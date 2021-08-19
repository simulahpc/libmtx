/* This file is part of libmtx.
 *
 * Copyright (C) 2021 James D. Trotter
 *
 * libmtx is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * libmtx is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with libmtx.  If not, see
 * <https://www.gnu.org/licenses/>.
 *
 * Authors: James D. Trotter <james@simula.no>
 * Last modified: 2021-08-19
 *
 * Communicating matrices in coordinate format between processes using
 * MPI.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/matrix/coordinate/mpi.h>
#include <libmtx/matrix/coordinate/data.h>

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>

/**
 * `mtx_matrix_coordinate_datatype()' creates a custom MPI data type
 * for sending or receiving data for a coordinate matrix with the
 * given field and precision.
 *
 * The user is responsible for calling `MPI_Type_free()' on the
 * returned datatype.
 */
static int mtx_matrix_coordinate_datatype(
    enum mtx_field field,
    enum mtx_precision precision,
    MPI_Datatype * datatype,
    int * mpierrcode)
{
    int num_elements;
    int block_lengths[4];
    MPI_Datatype element_types[4];
    MPI_Aint element_offsets[4];
    if (field == mtx_real) {
        if (precision == mtx_single) {
            num_elements = 3;
            element_types[0] = MPI_INT;
            block_lengths[0] = 1;
            element_offsets[0] =
                offsetof(struct mtx_matrix_coordinate_real_single, i);
            element_types[1] = MPI_INT;
            block_lengths[1] = 1;
            element_offsets[1] =
                offsetof(struct mtx_matrix_coordinate_real_single, j);
            element_types[2] = MPI_FLOAT;
            block_lengths[2] = 1;
            element_offsets[2] =
                offsetof(struct mtx_matrix_coordinate_real_single, a);
        } else if (precision == mtx_double) {
            num_elements = 3;
            element_types[0] = MPI_INT;
            block_lengths[0] = 1;
            element_offsets[0] =
                offsetof(struct mtx_matrix_coordinate_real_double, i);
            element_types[1] = MPI_INT;
            block_lengths[1] = 1;
            element_offsets[1] =
                offsetof(struct mtx_matrix_coordinate_real_double, j);
            element_types[2] = MPI_DOUBLE;
            block_lengths[2] = 1;
            element_offsets[2] =
                offsetof(struct mtx_matrix_coordinate_real_double, a);
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtx_complex) {
        if (precision == mtx_single) {
            num_elements = 3;
            element_types[0] = MPI_INT;
            block_lengths[0] = 1;
            element_offsets[0] =
                offsetof(struct mtx_matrix_coordinate_complex_single, i);
            element_types[1] = MPI_INT;
            block_lengths[1] = 1;
            element_offsets[1] =
                offsetof(struct mtx_matrix_coordinate_complex_single, j);
            element_types[2] = MPI_FLOAT;
            block_lengths[2] = 2;
            element_offsets[2] =
                offsetof(struct mtx_matrix_coordinate_complex_single, a);
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtx_integer) {
        if (precision == mtx_single) {
            num_elements = 3;
            element_types[0] = MPI_INT;
            block_lengths[0] = 1;
            element_offsets[0] =
                offsetof(struct mtx_matrix_coordinate_integer_single, i);
            element_types[1] = MPI_INT;
            block_lengths[1] = 1;
            element_offsets[1] =
                offsetof(struct mtx_matrix_coordinate_integer_single, j);
            element_types[2] = MPI_INT32_T;
            block_lengths[2] = 1;
            element_offsets[2] =
                offsetof(struct mtx_matrix_coordinate_integer_single, a);
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtx_pattern) {
        num_elements = 2;
        element_types[0] = MPI_INT;
        block_lengths[0] = 1;
        element_offsets[0] =
                offsetof(struct mtx_matrix_coordinate_pattern, i);
        element_types[1] = MPI_INT;
        block_lengths[1] = 1;
        element_offsets[1] =
                offsetof(struct mtx_matrix_coordinate_pattern, j);
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }

    /* Create an MPI data type for receiving nonzero data. */
    MPI_Datatype tmp_datatype;
    *mpierrcode = MPI_Type_create_struct(
        num_elements, block_lengths, element_offsets,
        element_types, &tmp_datatype);
    if (*mpierrcode)
        return MTX_ERR_MPI;

    /* Enable sending an array of the custom data type. */
    MPI_Aint lb, extent;
    *mpierrcode = MPI_Type_get_extent(tmp_datatype, &lb, &extent);
    if (*mpierrcode) {
        MPI_Type_free(&tmp_datatype);
        return MTX_ERR_MPI;
    }
    *mpierrcode = MPI_Type_create_resized(tmp_datatype, lb, extent, datatype);
    if (*mpierrcode) {
        MPI_Type_free(&tmp_datatype);
        return MTX_ERR_MPI;
    }
    *mpierrcode = MPI_Type_commit(datatype);
    if (*mpierrcode) {
        MPI_Type_free(datatype);
        MPI_Type_free(&tmp_datatype);
        return MTX_ERR_MPI;
    }

    MPI_Type_free(&tmp_datatype);
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_send()' sends a `struct
 * mtx_matrix_coordinate_data' to another MPI process.
 *
 * This is analogous to `MPI_Send()' and requires the receiving
 * process to perform a matching call to
 * `mtx_matrix_coordinate_recv()'.
 */
int mtx_matrix_coordinate_send(
    const struct mtx_matrix_coordinate_data * matrix_coordinate,
    int dest,
    int tag,
    MPI_Comm comm,
    int * mpierrcode)
{
    *mpierrcode = MPI_Send(
        &matrix_coordinate->field, 1, MPI_INT, dest, tag, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Send(
        &matrix_coordinate->precision, 1, MPI_INT, dest, tag, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Send(
        &matrix_coordinate->symmetry, 1, MPI_INT, dest, tag, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Send(
        &matrix_coordinate->triangle, 1, MPI_INT, dest, tag, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Send(
        &matrix_coordinate->sorting, 1, MPI_INT, dest, tag, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Send(
        &matrix_coordinate->assembly, 1, MPI_INT, dest, tag, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Send(
        &matrix_coordinate->num_rows, 1, MPI_INT, dest, tag, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Send(
        &matrix_coordinate->num_columns, 1, MPI_INT, dest, tag, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Send(
        &matrix_coordinate->size, 1, MPI_INT64_T, dest, tag, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;

    /* Get the data type. */
    MPI_Datatype datatype;
    int err = mtx_matrix_coordinate_datatype(
        matrix_coordinate->field,
        matrix_coordinate->precision,
        &datatype, mpierrcode);
    if (err)
        return err;

    if (matrix_coordinate->field == mtx_real) {
        if (matrix_coordinate->precision == mtx_single) {
            *mpierrcode = MPI_Send(
                matrix_coordinate->data.real_single,
                matrix_coordinate->size,
                datatype, dest, 0, comm);
            if (*mpierrcode) {
                MPI_Type_free(&datatype);
                return MTX_ERR_MPI;
            }
        } else if (matrix_coordinate->precision == mtx_double) {
            *mpierrcode = MPI_Send(
                matrix_coordinate->data.real_double,
                matrix_coordinate->size,
                datatype, dest, 0, comm);
            if (*mpierrcode) {
                MPI_Type_free(&datatype);
                return MTX_ERR_MPI;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (matrix_coordinate->field == mtx_complex) {
        if (matrix_coordinate->precision == mtx_single) {
            *mpierrcode = MPI_Send(
                matrix_coordinate->data.complex_single,
                matrix_coordinate->size,
                datatype, dest, 0, comm);
            if (*mpierrcode) {
                MPI_Type_free(&datatype);
                return MTX_ERR_MPI;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (matrix_coordinate->field == mtx_integer) {
        if (matrix_coordinate->precision == mtx_single) {
            *mpierrcode = MPI_Send(
                matrix_coordinate->data.integer_single,
                matrix_coordinate->size,
                datatype, dest, 0, comm);
            if (*mpierrcode) {
                MPI_Type_free(&datatype);
                return MTX_ERR_MPI;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (matrix_coordinate->field == mtx_pattern) {
        *mpierrcode = MPI_Send(
            matrix_coordinate->data.pattern,
            matrix_coordinate->size,
            datatype, dest, 0, comm);
        if (*mpierrcode) {
            MPI_Type_free(&datatype);
            return MTX_ERR_MPI;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_recv()' receives a `struct
 * mtx_matrix_coordinate_data' from another MPI process.
 *
 * This is analogous to `MPI_Recv()' and requires the sending process
 * to perform a matching call to `mtx_matrix_coordinate_send()'.
 */
int mtx_matrix_coordinate_recv(
    struct mtx_matrix_coordinate_data * matrix_coordinate,
    int source,
    int tag,
    MPI_Comm comm,
    int * mpierrcode)
{
    int err;
    enum mtx_field field;
    enum mtx_precision precision;
    enum mtx_symmetry symmetry;
    enum mtx_triangle triangle;
    enum mtx_sorting sorting;
    enum mtx_assembly assembly;
    int num_rows;
    int num_columns;
    int64_t size;

    *mpierrcode = MPI_Recv(
        &field, 1, MPI_INT, source, tag, comm,
        MPI_STATUS_IGNORE);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Recv(
        &precision, 1, MPI_INT, source, tag, comm,
        MPI_STATUS_IGNORE);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Recv(
        &symmetry, 1, MPI_INT, source, tag, comm,
        MPI_STATUS_IGNORE);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Recv(
        &triangle, 1, MPI_INT, source, tag, comm,
        MPI_STATUS_IGNORE);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Recv(
        &sorting, 1, MPI_INT, source, tag, comm,
        MPI_STATUS_IGNORE);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Recv(
        &assembly, 1, MPI_INT, source, tag, comm,
        MPI_STATUS_IGNORE);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Recv(
        &num_rows, 1, MPI_INT, source, tag, comm,
        MPI_STATUS_IGNORE);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Recv(
        &num_columns, 1, MPI_INT, source, tag, comm,
        MPI_STATUS_IGNORE);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Recv(
        &size, 1, MPI_INT64_T, source, tag, comm,
        MPI_STATUS_IGNORE);
    if (*mpierrcode)
        return MTX_ERR_MPI;

    /* Allocate storage for receiving the data. */
    err = mtx_matrix_coordinate_data_alloc(
        matrix_coordinate, field, precision,
        num_rows, num_columns, size);
    if (err)
        return err;
    matrix_coordinate->symmetry = symmetry;
    matrix_coordinate->triangle = triangle;
    matrix_coordinate->sorting = sorting;
    matrix_coordinate->assembly = assembly;

    /* Get the data type. */
    MPI_Datatype datatype;
    err = mtx_matrix_coordinate_datatype(
        field, precision,
        &datatype, mpierrcode);
    if (err) {
        mtx_matrix_coordinate_data_free(matrix_coordinate);
        return err;
    }

    if (matrix_coordinate->field == mtx_real) {
        if (matrix_coordinate->precision == mtx_single) {
            *mpierrcode = MPI_Recv(
                matrix_coordinate->data.real_single,
                matrix_coordinate->size,
                datatype, source, 0, comm,
                MPI_STATUS_IGNORE);
            if (*mpierrcode) {
                mtx_matrix_coordinate_data_free(matrix_coordinate);
                MPI_Type_free(&datatype);
                return MTX_ERR_MPI;
            }
        } else if (matrix_coordinate->precision == mtx_double) {
            *mpierrcode = MPI_Recv(
                matrix_coordinate->data.real_double,
                matrix_coordinate->size,
                datatype, source, 0, comm,
                MPI_STATUS_IGNORE);
            if (*mpierrcode) {
                mtx_matrix_coordinate_data_free(matrix_coordinate);
                MPI_Type_free(&datatype);
                return MTX_ERR_MPI;
            }
        } else {
            mtx_matrix_coordinate_data_free(matrix_coordinate);
            MPI_Type_free(&datatype);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (matrix_coordinate->field == mtx_complex) {
        if (matrix_coordinate->precision == mtx_single) {
            *mpierrcode = MPI_Recv(
                matrix_coordinate->data.complex_single,
                matrix_coordinate->size,
                datatype, source, 0, comm,
                MPI_STATUS_IGNORE);
            if (*mpierrcode) {
                mtx_matrix_coordinate_data_free(matrix_coordinate);
                MPI_Type_free(&datatype);
                return MTX_ERR_MPI;
            }
        } else {
            mtx_matrix_coordinate_data_free(matrix_coordinate);
            MPI_Type_free(&datatype);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (matrix_coordinate->field == mtx_integer) {
        if (matrix_coordinate->precision == mtx_single) {
            *mpierrcode = MPI_Recv(
                matrix_coordinate->data.integer_single,
                matrix_coordinate->size,
                datatype, source, 0, comm,
                MPI_STATUS_IGNORE);
            if (*mpierrcode) {
                mtx_matrix_coordinate_data_free(matrix_coordinate);
                MPI_Type_free(&datatype);
                return MTX_ERR_MPI;
            }
        } else {
            mtx_matrix_coordinate_data_free(matrix_coordinate);
            MPI_Type_free(&datatype);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (matrix_coordinate->field == mtx_pattern) {
        *mpierrcode = MPI_Recv(
            matrix_coordinate->data.pattern,
            matrix_coordinate->size,
            datatype, source, 0, comm,
            MPI_STATUS_IGNORE);
        if (*mpierrcode) {
            mtx_matrix_coordinate_data_free(matrix_coordinate);
            MPI_Type_free(&datatype);
            return MTX_ERR_MPI;
        }
    } else {
        mtx_matrix_coordinate_data_free(matrix_coordinate);
        MPI_Type_free(&datatype);
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    MPI_Type_free(&datatype);
    return MTX_SUCCESS;
}

/**
 * `mtx_matrix_coordinate_bcast()' broadcasts a `struct
 * mtx_matrix_coordinate_data' from an MPI root process to other
 * processes in a communicator.
 *
 * This is analogous to `MPI_Bcast()' and requires every process in
 * the communicator to perform matching calls to
 * `mtx_matrix_coordinate_bcast()'.
 */
int mtx_matrix_coordinate_bcast(
    struct mtx_matrix_coordinate_data * matrix_coordinate,
    int root,
    MPI_Comm comm,
    int * mpierrcode)
{
    int err;

    /* Get the MPI rank of the current process. */
    int rank;
    *mpierrcode = MPI_Comm_rank(comm, &rank);
    if (*mpierrcode)
        return MTX_ERR_MPI;

    enum mtx_field field;
    enum mtx_precision precision;
    enum mtx_symmetry symmetry;
    enum mtx_triangle triangle;
    enum mtx_sorting sorting;
    enum mtx_assembly assembly;
    int num_rows;
    int num_columns;
    int64_t size;

    if (rank == root) {
        field = matrix_coordinate->field;
        precision = matrix_coordinate->precision;
        symmetry = matrix_coordinate->symmetry;
        triangle = matrix_coordinate->triangle;
        sorting = matrix_coordinate->sorting;
        assembly = matrix_coordinate->assembly;
        num_rows = matrix_coordinate->num_rows;
        num_columns = matrix_coordinate->num_columns;
        size = matrix_coordinate->size;
    }

    *mpierrcode = MPI_Bcast(
        &field, 1, MPI_INT, root, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Bcast(
        &precision, 1, MPI_INT, root, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Bcast(
        &symmetry, 1, MPI_INT, root, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Bcast(
        &triangle, 1, MPI_INT, root, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Bcast(
        &sorting, 1, MPI_INT, root, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Bcast(
        &assembly, 1, MPI_INT, root, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Bcast(
        &num_rows, 1, MPI_INT, root, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Bcast(
        &num_columns, 1, MPI_INT, root, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;
    *mpierrcode = MPI_Bcast(
        &size, 1, MPI_INT64_T, root, comm);
    if (*mpierrcode)
        return MTX_ERR_MPI;

    /* Allocate storage for receiving the data. */
    if (rank != root) {
        err = mtx_matrix_coordinate_data_alloc(
            matrix_coordinate, field, precision,
            num_rows, num_columns, size);
        if (err)
            return err;
        matrix_coordinate->symmetry = symmetry;
        matrix_coordinate->triangle = triangle;
        matrix_coordinate->sorting = sorting;
        matrix_coordinate->assembly = assembly;
    }

    /* Get the data type. */
    MPI_Datatype datatype;
    err = mtx_matrix_coordinate_datatype(
        field, precision,
        &datatype, mpierrcode);
    if (err) {
        if (rank != root)
            mtx_matrix_coordinate_data_free(matrix_coordinate);
        return err;
    }

    if (matrix_coordinate->field == mtx_real) {
        if (matrix_coordinate->precision == mtx_single) {
            *mpierrcode = MPI_Bcast(
                matrix_coordinate->data.real_single,
                matrix_coordinate->size,
                datatype, root, comm);
            if (*mpierrcode) {
                if (rank != root)
                    mtx_matrix_coordinate_data_free(matrix_coordinate);
                MPI_Type_free(&datatype);
                return MTX_ERR_MPI;
            }
        } else if (matrix_coordinate->precision == mtx_double) {
            *mpierrcode = MPI_Bcast(
                matrix_coordinate->data.real_double,
                matrix_coordinate->size,
                datatype, root, comm);
            if (*mpierrcode) {
                if (rank != root)
                    mtx_matrix_coordinate_data_free(matrix_coordinate);
                MPI_Type_free(&datatype);
                return MTX_ERR_MPI;
            }
        } else {
            if (rank != root)
                mtx_matrix_coordinate_data_free(matrix_coordinate);
            MPI_Type_free(&datatype);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (matrix_coordinate->field == mtx_complex) {
        if (matrix_coordinate->precision == mtx_single) {
            *mpierrcode = MPI_Bcast(
                matrix_coordinate->data.complex_single,
                matrix_coordinate->size,
                datatype, root, comm);
            if (*mpierrcode) {
                if (rank != root)
                    mtx_matrix_coordinate_data_free(matrix_coordinate);
                MPI_Type_free(&datatype);
                return MTX_ERR_MPI;
            }
        } else {
            if (rank != root)
                mtx_matrix_coordinate_data_free(matrix_coordinate);
            MPI_Type_free(&datatype);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (matrix_coordinate->field == mtx_integer) {
        if (matrix_coordinate->precision == mtx_single) {
            *mpierrcode = MPI_Bcast(
                matrix_coordinate->data.integer_single,
                matrix_coordinate->size,
                datatype, root, comm);
            if (*mpierrcode) {
                if (rank != root)
                    mtx_matrix_coordinate_data_free(matrix_coordinate);
                MPI_Type_free(&datatype);
                return MTX_ERR_MPI;
            }
        } else {
            if (rank != root)
                mtx_matrix_coordinate_data_free(matrix_coordinate);
            MPI_Type_free(&datatype);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (matrix_coordinate->field == mtx_pattern) {
        *mpierrcode = MPI_Bcast(
            matrix_coordinate->data.pattern,
            matrix_coordinate->size,
            datatype, root, comm);
        if (*mpierrcode) {
            if (rank != root)
                mtx_matrix_coordinate_data_free(matrix_coordinate);
            MPI_Type_free(&datatype);
            return MTX_ERR_MPI;
        }
    } else {
        if (rank != root)
            mtx_matrix_coordinate_data_free(matrix_coordinate);
        MPI_Type_free(&datatype);
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    MPI_Type_free(&datatype);
    return MTX_SUCCESS;
}
#endif
