/* This file is part of libmtx.
 *
 * Copyright (C) 2022 James D. Trotter
 *
 * libmtx is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libmtx is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with libmtx.  If not, see <https://www.gnu.org/licenses/>.
 *
 * Authors: James D. Trotter <james@simula.no>
 * Last modified: 2022-01-05
 *
 * Matrix Market file headers.
 */

#ifndef LIBMTX_MTXFILE_HEADER_H
#define LIBMTX_MTXFILE_HEADER_H

#include <libmtx/libmtx-config.h>

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>
#endif

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <stddef.h>
#include <stdio.h>

struct mtxdisterror;

/*
 * Matrix Market header data types.
 */

/**
 * ‘mtxfileobject’ is used to enumerate different kinds of Matrix
 * Market objects.
 */
enum mtxfileobject
{
    mtxfile_matrix,
    mtxfile_vector
};

/**
 * ‘mtxfileobject_str()’ is a string representing the Matrix Market
 * object type.
 */
const char * mtxfileobject_str(
    enum mtxfileobject object);

/**
 * ‘mtxfileobject_parse()’ parses a string containing the ‘object’ of
 * a Matrix Market file format header.
 *
 * ‘valid_delimiters’ is either ‘NULL’, in which case it is ignored,
 * or it is a string of characters considered to be valid delimiters
 * for the parsed string.  That is, if there are any remaining,
 * non-NULL characters after parsing, then then the next character is
 * searched for in ‘valid_delimiters’.  If the character is found,
 * then the parsing succeeds and the final delimiter character is
 * consumed by the parser. Otherwise, the parsing fails with an error.
 *
 * If ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’
 * points to the first character beyond the characters that were
 * consumed during parsing.
 *
 * On success, ‘mtxfileobject_parse()’ returns ‘MTX_SUCCESS’ and
 * ‘object’ is set according to the parsed string and ‘bytes_read’ is
 * set to the number of bytes that were consumed by the parser.
 * Otherwise, an error code is returned.
 */
int mtxfileobject_parse(
    enum mtxfileobject * object,
    int64_t * bytes_read,
    const char ** endptr,
    const char * s,
    const char * valid_delimiters);

/**
 * ‘mtxfileformat’ is used to enumerate different kinds of Matrix
 * Market formats.
 */
enum mtxfileformat
{
    mtxfile_array,     /* array format for dense matrices and vectors */
    mtxfile_coordinate /* coordinate format for sparse matrices and vectors */
};

/**
 * ‘mtxfileformat_str()’ is a string representing the Matrix Market
 * format type.
 */
const char * mtxfileformat_str(
    enum mtxfileformat format);

/**
 * ‘mtxfileformat_parse()’ parses a string containing the ‘format’ of
 * a Matrix Market file format header.
 *
 * ‘valid_delimiters’ is either ‘NULL’, in which case it is ignored,
 * or it is a string of characters considered to be valid delimiters
 * for the parsed string.  That is, if there are any remaining,
 * non-NULL characters after parsing, then then the next character is
 * searched for in ‘valid_delimiters’.  If the character is found,
 * then the parsing succeeds and the final delimiter character is
 * consumed by the parser. Otherwise, the parsing fails with an error.
 *
 * If ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’
 * points to the first character beyond the characters that were
 * consumed during parsing.
 *
 * On success, ‘mtxfileformat_parse()’ returns ‘MTX_SUCCESS’ and
 * ‘format’ is set according to the parsed string and ‘bytes_read’ is
 * set to the number of bytes that were consumed by the parser.
 * Otherwise, an error code is returned.
 */
int mtxfileformat_parse(
    enum mtxfileformat * format,
    int64_t * bytes_read,
    const char ** endptr,
    const char * s,
    const char * valid_delimiters);

/**
 * ‘mtxfilefield’ is used to enumerate different kinds of fields for
 * matrix values in Matrix Market files.
 */
enum mtxfilefield
{
    mtxfile_real,    /* real, floating point coefficients */
    mtxfile_complex, /* complex, floating point coefficients */
    mtxfile_integer, /* integer coefficients */
    mtxfile_pattern  /* boolean coefficients (sparsity pattern) */
};

/**
 * ‘mtxfilefield_str()’ is a string representing the Matrix Market
 * field type.
 */
const char * mtxfilefield_str(
    enum mtxfilefield field);

/**
 * ‘mtxfilefield_parse()’ parses a string containing the ‘field’ of a
 * Matrix Market file format header.
 *
 * ‘valid_delimiters’ is either ‘NULL’, in which case it is ignored,
 * or it is a string of characters considered to be valid delimiters
 * for the parsed string.  That is, if there are any remaining,
 * non-NULL characters after parsing, then then the next character is
 * searched for in ‘valid_delimiters’.  If the character is found,
 * then the parsing succeeds and the final delimiter character is
 * consumed by the parser. Otherwise, the parsing fails with an error.
 *
 * If ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’
 * points to the first character beyond the characters that were
 * consumed during parsing.
 *
 * On success, ‘mtxfilefield_parse()’ returns ‘MTX_SUCCESS’ and
 * ‘field’ is set according to the parsed string and ‘bytes_read’ is
 * set to the number of bytes that were consumed by the parser.
 * Otherwise, an error code is returned.
 */
int mtxfilefield_parse(
    enum mtxfilefield * field,
    int64_t * bytes_read,
    const char ** endptr,
    const char * s,
    const char * valid_delimiters);

/**
 * ‘mtxfilesymmetry’ is used to enumerate different kinds of symmetry
 * for matrices in Matrix Market format.
 */
enum mtxfilesymmetry
{
    mtxfile_general,        /* general, non-symmetric matrix */
    mtxfile_symmetric,      /* symmetric matrix */
    mtxfile_skew_symmetric, /* skew-symmetric matrix */
    mtxfile_hermitian       /* Hermitian matrix */
};

/**
 * ‘mtxfilesymmetry_str()’ is a string representing the Matrix Market
 * symmetry type.
 */
const char * mtxfilesymmetry_str(
    enum mtxfilesymmetry symmetry);

/**
 * ‘mtxfilesymmetry_parse()’ parses a string containing the
 * ‘symmetry’ of a Matrix Market file format header.
 *
 * ‘valid_delimiters’ is either ‘NULL’, in which case it is ignored,
 * or it is a string of characters considered to be valid delimiters
 * for the parsed string.  That is, if there are any remaining,
 * non-NULL characters after parsing, then then the next character is
 * searched for in ‘valid_delimiters’.  If the character is found,
 * then the parsing succeeds and the final delimiter character is
 * consumed by the parser. Otherwise, the parsing fails with an error.
 *
 * If ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’
 * points to the first character beyond the characters that were
 * consumed during parsing.
 *
 * On success, ‘mtxfilesymmetry_parse()’ returns ‘MTX_SUCCESS’ and
 * ‘symmetry’ is set according to the parsed string and ‘bytes_read’
 * is set to the number of bytes that were consumed by the parser.
 * Otherwise, an error code is returned.
 */
int mtxfilesymmetry_parse(
    enum mtxfilesymmetry * symmetry,
    int64_t * bytes_read,
    const char ** endptr,
    const char * s,
    const char * valid_delimiters);

/*
 * Matrix Market header.
 */

/**
 * ‘mtxfileheader’ represents the header line of a Matrix Market
 * file.
 */
struct mtxfileheader
{
    /**
     * ‘object’ is the type of Matrix Market object: ‘matrix’ or
     * ‘vector’.
     */
    enum mtxfileobject object;

    /**
     * ‘format’ is the matrix format: ‘coordinate’ or ‘array’.
     */
    enum mtxfileformat format;

    /**
     * ‘field’ is the matrix field: ‘real’, ‘complex’, ‘integer’ or
     * ‘pattern’.
     */
    enum mtxfilefield field;

    /**
     * ‘symmetry’ is the matrix symmetry: ‘general’, ‘symmetric’,
     * ‘skew-symmetric’, or ‘hermitian’.
     *
     * Note that if ‘symmetry’ is ‘symmetric’, ‘skew-symmetric’ or
     * ‘hermitian’, then the matrix must be square, so that ‘num_rows’
     * is equal to ‘num_columns’.
     */
    enum mtxfilesymmetry symmetry;
};

/**
 * ‘mtxfileheader_parse()’ parses a string containing the header line
 * for a file in Matrix Market format.
 *
 * If ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’
 * points to the first character beyond the characters that were
 * consumed during parsing.
 *
 * On success, ‘mtxfileheader_parse()’ returns ‘MTX_SUCCESS’ and the
 * ‘object’, ‘format’, ‘field’ and ‘symmetry’ fields of the header
 * will be set according to the contents of the parsed Matrix Market
 * header.  Otherwise, an appropriate error code is returned if the
 * input is not a valid Matrix Market header.
 */
int mtxfileheader_parse(
    struct mtxfileheader * header,
    int64_t * bytes_read,
    const char ** endptr,
    const char * s);

/**
 * ‘mtxfileheader_copy()’ copies a Matrix Market header.
 */
int mtxfileheader_copy(
    struct mtxfileheader * dst,
    const struct mtxfileheader * src);

/*
 * I/O functions
 */

/**
 * ‘mtxfileheader_fread()’ reads a Matrix Market header from a
 * stream.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxfileheader_fread(
    struct mtxfileheader * header,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf);

#ifdef LIBMTX_HAVE_LIBZ
/**
 * ‘mtxfileheader_gzread()’ reads a Matrix Market header from a
 * gzip-compressed stream.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxfileheader_gzread(
    struct mtxfileheader * header,
    gzFile f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf);
#endif

/**
 * ‘mtxfileheader_fwrite()’ writes the header line of a Matrix Market
 * file to a stream.
 *
 * If it is not ‘NULL’, then the number of bytes written to the stream
 * is returned in ‘bytes_written’.
 */
int mtxfileheader_fwrite(
    const struct mtxfileheader * header,
    FILE * f,
    int64_t * bytes_written);

#ifdef LIBMTX_HAVE_LIBZ
/**
 * ‘mtxfileheader_gzwrite()’ writes the header line of a Matrix
 * Market file to a gzip-compressed stream.
 *
 * If it is not ‘NULL’, then the number of bytes written to the stream
 * is returned in ‘bytes_written’.
 */
int mtxfileheader_gzwrite(
    const struct mtxfileheader * header,
    gzFile f,
    int64_t * bytes_written);
#endif

/*
 * MPI functions
 */

#ifdef LIBMTX_HAVE_MPI
/**
 * ‘mtxfileheader_send()’ sends a Matrix Market header to another MPI
 * process.
 *
 * This is analogous to ‘MPI_Send()’ and requires the receiving
 * process to perform a matching call to ‘mtxfileheader_recv()’.
 */
int mtxfileheader_send(
    const struct mtxfileheader * header,
    int dest,
    int tag,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxfileheader_recv()’ receives a Matrix Market header from
 * another MPI process.
 *
 * This is analogous to ‘MPI_Recv()’ and requires the sending process
 * to perform a matching call to ‘mtxfileheader_send()’.
 */
int mtxfileheader_recv(
    struct mtxfileheader * header,
    int source,
    int tag,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxfileheader_bcast()’ broadcasts a Matrix Market header from an
 * MPI root process to other processes in a communicator.
 *
 * This is analogous to ‘MPI_Bcast()’ and requires every process in
 * the communicator to perform matching calls to
 * ‘mtxfileheader_bcast()’.
 */
int mtxfileheader_bcast(
    struct mtxfileheader * header,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxfileheader_gather()’ gathers Matrix Market headers onto an MPI
 * root process from other processes in a communicator.
 *
 * This is analogous to ‘MPI_Gather()’ and requires every process in
 * the communicator to perform matching calls to
 * ‘mtxfileheader_gather()’.
 */
int mtxfileheader_gather(
    const struct mtxfileheader * sendheader,
    struct mtxfileheader * recvheaders,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxfileheader_allgather()’ gathers Matrix Market headers onto
 * every MPI process from other processes in a communicator.
 *
 * This is analogous to ‘MPI_Allgather()’ and requires every process
 * in the communicator to perform matching calls to this function.
 */
int mtxfileheader_allgather(
    const struct mtxfileheader * sendheader,
    struct mtxfileheader * recvheaders,
    MPI_Comm comm,
    struct mtxdisterror * disterr);
#endif

#endif
