#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define CGLTF_IMPLEMENTATION
#include "cgltf.h"

// dr_mp3 v0.7.4 (public domain / MIT-0, see LICENSES.md and end of the file):
// single-header MP3 decoder; the implementation is compiled exactly once, here.
#define DR_MP3_IMPLEMENTATION
#include "dr_mp3.h"

// stb_vorbis v1.22 (public domain / MIT, see LICENSES.md and end of the file):
// the file is both header and implementation, so it is included directly.
// Zig-side @cImport sees only declarations via STB_VORBIS_HEADER_ONLY
// (src/agate/audio/decode.zig); compiling it here keeps that translation
// unit declarations-only.
#include "stb_vorbis.c"

// meshoptimizer v1.2 (MIT, see meshopt/LICENSE.md and LICENSES.md): decoder
// only. cgltf parses EXT_meshopt_compression metadata but leaves the actual
// decompression to the host (agate_cgltf_decode_meshopt below); the decoder
// sources are vendored verbatim and compiled as C++ from build.zig.
#include "meshopt/meshoptimizer.h"

cgltf_result agate_cgltf_decode_meshopt(const cgltf_options* options, cgltf_data* data)
{
	void* (*mem_alloc)(void*, cgltf_size) = options->memory.alloc_func ? options->memory.alloc_func : &cgltf_default_alloc;
	void (*mem_free)(void*, void*) = options->memory.free_func ? options->memory.free_func : &cgltf_default_free;

	for (cgltf_size i = 0; i < data->buffer_views_count; ++i)
	{
		cgltf_buffer_view* view = &data->buffer_views[i];
		if (!view->has_meshopt_compression)
		{
			continue;
		}

		const cgltf_meshopt_compression* mc = &view->meshopt_compression;
		if (!mc->buffer || !mc->buffer->data)
		{
			return cgltf_result_invalid_gltf;
		}
		if (mc->offset > mc->buffer->size || mc->size > mc->buffer->size - mc->offset)
		{
			return cgltf_result_data_too_short;
		}

		if (mc->stride != 0 && mc->count > ((cgltf_size)-1) / mc->stride)
		{
			return cgltf_result_invalid_gltf;
		}

		const unsigned char* source = (const unsigned char*)mc->buffer->data + mc->offset;
		void* decoded = mem_alloc(options->memory.user_data, mc->count * mc->stride);
		if (!decoded)
		{
			return cgltf_result_out_of_memory;
		}

		// Returns 0 on success; safe for untrusted input (bounded decode).
		int err;
		switch (mc->mode)
		{
		case cgltf_meshopt_compression_mode_attributes:
			err = meshopt_decodeVertexBuffer(decoded, mc->count, mc->stride, source, mc->size);
			break;
		case cgltf_meshopt_compression_mode_triangles:
			err = meshopt_decodeIndexBuffer(decoded, mc->count, mc->stride, source, mc->size);
			break;
		case cgltf_meshopt_compression_mode_indices:
			err = meshopt_decodeIndexSequence(decoded, mc->count, mc->stride, source, mc->size);
			break;
		default:
			err = -1;
			break;
		}

		if (err != 0)
		{
			mem_free(options->memory.user_data, decoded);
			return cgltf_result_invalid_gltf;
		}

		// Attribute filters run after the lossless decode (EXT spec order);
		// index modes reject non-none filters at parse time already.
		switch (mc->filter)
		{
		case cgltf_meshopt_compression_filter_octahedral:
			meshopt_decodeFilterOct(decoded, mc->count, mc->stride);
			break;
		case cgltf_meshopt_compression_filter_quaternion:
			meshopt_decodeFilterQuat(decoded, mc->count, mc->stride);
			break;
		case cgltf_meshopt_compression_filter_exponential:
			meshopt_decodeFilterExp(decoded, mc->count, mc->stride);
			break;
		case cgltf_meshopt_compression_filter_color:
			meshopt_decodeFilterColor(decoded, mc->count, mc->stride);
			break;
		default:
			break;
		}

		// The decoded blob replaces the view contents. cgltf_free releases
		// buffer_views[i].data with memory.free_func, matching our alloc.
		view->data = decoded;
		// glTF byteStride absent means tightly packed; the decoded blob is
		// exactly count*stride, so accessors must derive their stride from
		// the extension, not from element size.
		if (view->stride == 0)
		{
			view->stride = mc->stride;
		}
	}

	return cgltf_result_success;
}

