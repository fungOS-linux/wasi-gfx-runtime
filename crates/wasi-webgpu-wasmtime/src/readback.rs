//! Host-side texture readback used by presentation bridges.

use std::sync::Arc;

use wasmtime::component::{Resource, ResourceTable};
use wgpu_core::device::HostMap;
use wgpu_core::global::Global;
use wgpu_core::resource::{BufferDescriptor, BufferMapOperation};
use wgpu_types::{
    BufferUsages, Extent3d, Origin3d, PollType, TexelCopyBufferInfo, TexelCopyBufferLayout,
    TexelCopyTextureInfo, TextureAspect, TextureFormat,
};

use crate::wasi::webgpu::webgpu::GpuTexture;
use crate::TextureInfo;

/// An RGBA8 image copied out of a guest `wasi:webgpu` texture.
pub struct TextureImage {
    pub width: u32,
    pub height: u32,
    /// Always tightly packed, top-to-bottom RGBA8.
    pub rgba: Vec<u8>,
}

/// Copy a 2D guest texture into a CPU RGBA8 buffer.
///
/// The texture must have been created by `wasi:webgpu`'s `create-texture` and
/// carry `copy-src`. Only the formats Canvas can composite directly are
/// accepted; anything else is rejected instead of silently reinterpreted.
pub fn read_texture_rgba(
    instance: &Arc<Global>,
    table: &ResourceTable,
    texture: Resource<GpuTexture>,
) -> wasmtime::Result<TextureImage> {
    let texture = table.get(&texture)?;
    let info = texture
        .info()
        .ok_or_else(|| wasmtime::Error::msg("texture has no guest descriptor"))?;
    if info.width == 0 || info.height == 0 {
        wasmtime::bail!("cannot present an empty texture");
    }
    if !info.usage.contains(wgpu_types::TextureUsages::COPY_SRC) {
        wasmtime::bail!("presented texture must have copy-src usage");
    }

    let (bytes_per_pixel, swap_rb) = format_layout(info)?;
    let row_bytes = info.width * bytes_per_pixel;
    let padded_row_bytes = row_bytes.div_ceil(256) * 256;
    let buffer_size = u64::from(padded_row_bytes) * u64::from(info.height);

    let (buffer, err) = instance.device_create_buffer(
        texture.device_id,
        &BufferDescriptor {
            label: Some(std::borrow::Cow::Borrowed("soil canvas readback")),
            size: buffer_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
        None,
    );
    if let Some(err) = err {
        wasmtime::bail!("create readback buffer: {err}");
    }

    let (encoder, err) = instance.device_create_command_encoder(
        texture.device_id,
        &wgpu_types::CommandEncoderDescriptor {
            label: Some(std::borrow::Cow::Borrowed("soil canvas readback")),
        },
        None,
    );
    if let Some(err) = err {
        instance.buffer_destroy(buffer);
        wasmtime::bail!("create readback encoder: {err}");
    }

    let copy = instance.command_encoder_copy_texture_to_buffer(
        encoder,
        &TexelCopyTextureInfo {
            texture: texture.texture_id,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        &TexelCopyBufferInfo {
            buffer,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row_bytes),
                rows_per_image: Some(info.height),
            },
        },
        &Extent3d {
            width: info.width,
            height: info.height,
            depth_or_array_layers: 1,
        },
    );
    if let Err(err) = copy {
        instance.buffer_destroy(buffer);
        instance.command_encoder_drop(encoder);
        wasmtime::bail!("copy texture to readback buffer: {err}");
    }

    let (command_buffer, err) = instance.command_encoder_finish(
        encoder,
        &wgpu_types::CommandBufferDescriptor { label: None },
        None,
    );
    if let Some((_, err)) = err {
        instance.buffer_destroy(buffer);
        wasmtime::bail!("finish readback encoder: {err}");
    }

    if let Err(err) = instance.queue_submit(texture.queue_id, &[command_buffer]) {
        instance.buffer_destroy(buffer);
        wasmtime::bail!("submit readback: {err:?}");
    }

    let operation = BufferMapOperation {
        host: HostMap::Read,
        callback: Some(Box::new(|_| {})),
    };
    if let Err(err) = instance.buffer_map_async(buffer, 0, Some(buffer_size), operation) {
        instance.buffer_destroy(buffer);
        wasmtime::bail!("map readback buffer: {err}");
    }
    if let Err(err) = instance.device_poll(texture.device_id, PollType::wait_indefinitely()) {
        instance.buffer_unmap(buffer)?;
        instance.buffer_destroy(buffer);
        wasmtime::bail!("wait for readback: {err}");
    }

    let (pointer, len) = match instance.buffer_get_mapped_range(buffer, 0, Some(buffer_size)) {
        Ok(range) => range,
        Err(err) => {
            instance.buffer_destroy(buffer);
            return Err(wasmtime::Error::msg(format!(
                "get mapped readback range: {err}"
            )));
        }
    };
    // SAFETY: wgpu just handed us a mapped range of exactly `len` bytes that
    // remains valid until the buffer is unmapped.
    let mapped = unsafe { std::slice::from_raw_parts(pointer.as_ptr(), len as usize) };
    let mut rgba = Vec::with_capacity((row_bytes * info.height) as usize);
    for row in 0..info.height as usize {
        let start = row * padded_row_bytes as usize;
        let row = &mapped[start..start + row_bytes as usize];
        if swap_rb {
            for pixel in row.as_chunks::<4>().0 {
                rgba.extend_from_slice(&[pixel[2], pixel[1], pixel[0], pixel[3]]);
            }
        } else {
            rgba.extend_from_slice(row);
        }
    }

    instance.buffer_unmap(buffer)?;
    instance.buffer_destroy(buffer);

    Ok(TextureImage {
        width: info.width,
        height: info.height,
        rgba,
    })
}

fn format_layout(info: TextureInfo) -> wasmtime::Result<(u32, bool)> {
    Ok(match info.format {
        TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb => (4, false),
        TextureFormat::Bgra8Unorm | TextureFormat::Bgra8UnormSrgb => (4, true),
        other => wasmtime::bail!("unsupported canvas texture format: {other:?}"),
    })
}
