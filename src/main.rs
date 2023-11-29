use assets_manager::{asset::Gltf, AssetCache};
use frenderer::{
    input,
    meshes::MeshGroup,
    wgpu::{self, Color},
    Camera3D, PollsterRuntime, Renderer, Transform3D,
};
//use glam::*;
use glam::{Vec3, Quat, EulerRot};
use rand::{rngs::ThreadRng, Rng};
use ultraviolet::Rotor3;
use ultraviolet::*;

use crate::camera::Camera;

// to run, do:
// cargo run --bin test-gltf

mod camera;

use std::f32::consts::PI;

const DT: f32 = 1.0 / 60.0;
const BACKGROUND_COLOR: wgpu::Color = Color {
    r: 0.0 / 255.0,
    g: 200.0 / 255.0,
    b: 255.0 / 255.0,
    a: 1.0,
};

const PLAYER_SPEED: f32 = 100.0;

/* create mesh group with multiple textures
 */
fn create_mesh_flatten_multiple(
    cache: &AssetCache,
    frend: &mut Renderer<PollsterRuntime>,
    sprite: &str,
    instance_count: u32,
) -> MeshGroup {
    let sprite_gltf = cache.load::<assets_manager::asset::Gltf>(&sprite).unwrap();
    let asset = sprite_gltf.read();

    // add lines for asset
    let mut mats: Vec<_> = asset
        .document
        .materials()
        .map(|m| m.pbr_metallic_roughness().base_color_factor())
        .collect();
    if mats.is_empty() {
        mats.push([1.0, 0.0, 0.0, 1.0]);
    }
    let mut verts = Vec::with_capacity(1024);
    let mut indices = Vec::with_capacity(1024);
    let mut entries = Vec::with_capacity(1);
    for mesh in asset.document.meshes() {
        let mut entry = frenderer::meshes::MeshEntry {
            instance_count,
            submeshes: Vec::with_capacity(1),
        };
        for prim in mesh.primitives() {
            let reader = prim.reader(|b| Some(asset.get_buffer(&b)));
            let vtx_old_len = verts.len();
            assert_eq!(prim.mode(), gltf::mesh::Mode::Triangles);
            verts.extend(reader.read_positions().unwrap().map(|position| {
                frenderer::meshes::FlatVertex::new(
                    position,
                    prim.material().index().unwrap_or(0) as u32,
                )
            }));
            let idx_old_len = indices.len();
            match reader.read_indices() {
                None => indices.extend(0..(verts.len() - vtx_old_len) as u32),
                Some(index_reader) => indices.extend(index_reader.into_u32()),
            };
            entry.submeshes.push(frenderer::meshes::SubmeshData {
                indices: idx_old_len as u32..(indices.len() as u32),
                vertex_base: vtx_old_len as i32,
            })
        }
        assert!(!entry.submeshes.is_empty());
        entries.push(dbg!(entry));
    }
    let sprite_mesh = frend
        .flats
        .add_mesh_group(&frend.gpu, &mats, verts, indices, entries);
    return sprite_mesh
}

/*
create mesh for specified sprite
@params:
    - sprite (GameObject): has information to create sprite texture
 */
fn create_mesh_single_texture(
    cache: &AssetCache,
    frend: &mut Renderer<PollsterRuntime>,
    sprite: &str,
) -> MeshGroup {
    let sprite_gltf = cache.load::<assets_manager::asset::Gltf>(&sprite).unwrap();
    let game_sprite = sprite_gltf.read();
    let sprite_img = game_sprite.get_image_by_index(0); // game sprite is asset in frenderermain
    let sprite_tex = frend.gpu.create_array_texture(
        &[&sprite_img.to_rgba8()],
        frenderer::wgpu::TextureFormat::Rgba8Unorm,
        (sprite_img.width(), sprite_img.height()),
        Some("texture"), // string concatenation
    );

    const COUNT: usize = 10;
    let prim = game_sprite
        .document
        .meshes()
        .next()
        .unwrap()
        .primitives()
        .next()
        .unwrap();
    let reader = prim.reader(|b| Some(game_sprite.get_buffer_by_index(b.index())));

    let verts: Vec<_> = reader
        .read_positions()
        .unwrap()
        .zip(reader.read_tex_coords(0).unwrap().into_f32())
        .map(|(position, uv)| frenderer::meshes::Vertex::new(position, uv, 0))
        .collect();
    let vert_count = verts.len();

    let sprite_mesh = frend.meshes.add_mesh_group(
        &frend.gpu,
        &sprite_tex,
        verts,
        (0..vert_count as u32).collect(),
        vec![frenderer::meshes::MeshEntry {
            instance_count: COUNT as u32,
            submeshes: vec![frenderer::meshes::SubmeshEntry {
                vertex_base: 0,
                indices: 0..vert_count as u32,
            }],
        }],
    );
    return sprite_mesh;
}

/* perform transformation on mesh
@params:
- flat (bool): flattened mesh or not (i.e. MeshGroup was created from create_mesh_flatten_multiple)
*/
fn transform_mesh(
    rng: &mut ThreadRng,
    frend: &mut Renderer<PollsterRuntime>,
    mesh: MeshGroup,
    flat: bool,
    scale_start: f32,
    scale_end: f32,
) {
    if !flat {
        for trf in frend.meshes.get_meshes_mut(mesh, 0) {
            *trf = Transform3D {
                translation: Vec3 {
                    x: rng.gen_range(-400.0..400.0),
                    y: rng.gen_range(-300.0..300.0),
                    z: rng.gen_range(-500.0..-200.0),
                }
                .into(),
                rotation: Quat::from_euler(
                    EulerRot::XYZ,
                    rng.gen_range(0.0..std::f32::consts::TAU),
                    rng.gen_range(0.0..std::f32::consts::TAU),
                    rng.gen_range(0.0..std::f32::consts::TAU),
                )
                .into(),
                scale: rng.gen_range(scale_start..scale_end),
            };
        }
        frend.meshes.upload_meshes_group(&frend.gpu, mesh);
    } else {
        for i in 0..frend.flats.mesh_count(mesh) {
            for trf in frend.flats.get_meshes_mut(mesh, i) {
                *trf = Transform3D {
                    translation: Vec3 {
                        x: rng.gen_range(-400.0..400.0),
                        y: rng.gen_range(-300.0..300.0),
                        z: rng.gen_range(-500.0..-100.0),
                    }
                    .into(),
                    rotation: Rotor3::from_euler_angles(
                        rng.gen_range(0.0..std::f32::consts::TAU),
                        rng.gen_range(0.0..std::f32::consts::TAU),
                        rng.gen_range(0.0..std::f32::consts::TAU),
                    )
                    .into_quaternion_array(),
                    scale: rng.gen_range(scale_start..scale_end),
                };
            }
            frend.flats.upload_meshes_group(&frend.gpu, mesh);
        }
    }
}

/* spawns a mesh at a given point
@params:
- flat (bool): flattened mesh or not (i.e. MeshGroup was created from create_mesh_flatten_multiple)
*/
fn spawn(
    frend: &mut Renderer<PollsterRuntime>,
    mesh: MeshGroup,
    flat: bool,
    scale: f32,
    x: f32,
    y: f32,
    z: f32,
    angle_a: f32,
    angle_b: f32,
    angle_c: f32
) {
    if !flat {
        for trf in frend.meshes.get_meshes_mut(mesh, 0) {
            *trf = Transform3D {
                translation: Vec3 {
                    x: x,
                    y: y,
                    z: z,
                }
                .into(),
                rotation: Quat::from_euler(
                    EulerRot::XYZ,
                    angle_a,
                    angle_b,
                    angle_c,
                )
                .into(),
                scale: scale,
            };
        }
        frend.meshes.upload_meshes_group(&frend.gpu, mesh);
    } else {
        for i in 0..frend.flats.mesh_count(mesh) {
            for trf in frend.flats.get_meshes_mut(mesh, i) {
                *trf = Transform3D {
                    translation: Vec3 {
                        x: x,
                        y: y,
                        z: z,
                    }
                    .into(),
                    rotation: Rotor3::from_euler_angles(
                        angle_a,
                        angle_b,
                        angle_c,
                    )
                    .into_quaternion_array(),
                    scale: scale
                };
            }
            frend.flats.upload_meshes_group(&frend.gpu, mesh);
        }
    }
}

fn main() {
    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::Window::new(&event_loop).unwrap();
    #[cfg(not(target_arch = "wasm32"))]
    let source = assets_manager::source::FileSystem::new("content").unwrap();
    #[cfg(target_arch = "wasm32")]
    let source = assets_manager::source::Embedded::from(source::embed!("content"));
    let cache = assets_manager::AssetCache::with_source(source);
    let mut frend = frenderer::with_default_runtime(&window);
    let mut input = input::Input::default();

    let mut camera = Camera3D {
        translation: Vec3 {
            x: 0.0,
            y: 25.0,
            z: -100.0,
        }
        .into(),
        rotation: Quat::from_rotation_y(0.0).into(),
        // 90 degrees is typical
        fov: std::f32::consts::FRAC_PI_2,
        near: 10.0,
        far: 1000.0,
        aspect: 1024.0 / 768.0,
    };
    frend.meshes.set_camera(&frend.gpu, camera);
    frend.flats.set_camera(&frend.gpu, camera);

    let mut player_transform: Transform3D = Transform3D {
        translation: (camera.translation),
        scale: (1.0),
        rotation: (camera.rotation),
    };

    let mut fpcamera: Camera = Camera {
        pitch: 0.0,
        yaw: 0.0,
        player_pos: player_transform.translation.into(),
        player_rot: Quat::from_array(player_transform.rotation),
    };

    let mut rng = rand::thread_rng();

    // defines meshes using create_mesh_single_texture or create_gltf_flatten_multiple
    let fox_mesh = create_mesh_single_texture(&cache, &mut frend, "Fox");
    let raccoon_mesh = create_mesh_flatten_multiple(&cache, &mut frend, "scene", 10);
    let world_mesh = create_mesh_flatten_multiple(&cache, &mut frend, "GraceLiTrial", 1);

    // apply transformations
    transform_mesh(&mut rng, &mut frend, fox_mesh, false, 0.5, 1.0);
    transform_mesh(&mut rng, &mut frend, raccoon_mesh, true, 12.0, 20.0);
    // frend.meshes.upload_meshes_group(&frend.gpu, world_mesh);
    //transform_mesh(&mut rng, &mut frend, world_mesh, true, 50.0, 100.0);
    spawn(&mut frend, world_mesh, true, 13.0, 0.0, 0.0, 0.0, 0.0, PI, 0.0);
 
    const DT_FUDGE_AMOUNT: f32 = 0.0002;
    const DT_MAX: f32 = DT * 5.0;
    const TIME_SNAPS: [f32; 5] = [15.0, 30.0, 60.0, 120.0, 144.0];
    let mut acc = 0.0;
    let mut now = std::time::Instant::now();
    event_loop.run(move |event, _, control_flow| {
        use winit::event::{Event, WindowEvent};
        control_flow.set_poll();
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = winit::event_loop::ControlFlow::Exit;
            }
            Event::MainEventsCleared => {
                // compute elapsed time since last frame
                let mut elapsed = now.elapsed().as_secs_f32();
                //println!("{elapsed}");
                // snap time to nearby vsync framerate
                TIME_SNAPS.iter().for_each(|s| {
                    if (elapsed - 1.0 / s).abs() < DT_FUDGE_AMOUNT {
                        elapsed = 1.0 / s;
                    }
                });
                // Death spiral prevention
                if elapsed > DT_MAX {
                    acc = 0.0;
                    elapsed = DT;
                }
                acc += elapsed;
                now = std::time::Instant::now();
                // While we have time to spend
                while acc >= DT {
                    // simulate a frame
                    acc -= DT;
                    // rotate every fox a random amount
                    /*
                    for trf in frend.meshes.get_meshes_mut(fox_mesh, 0) {
                        trf.rotation = (Quat::from_array(trf.rotation)
                            * Quat::from_euler(
                                EulerRot::XYZ,
                                rng.gen_range(0.0..(std::f32::consts::TAU * DT)),
                                rng.gen_range(0.0..(std::f32::consts::TAU * DT)),
                                rng.gen_range(0.0..(std::f32::consts::TAU * DT)),
                           ))
                        .into();
                    }
                    camera.translation[2] -= 100.0 * DT;
                    */
                    let (mx, _my): (f32, f32) = input.mouse_delta().into();
                    // need to make rot into a quaternion
                    
                    let mut rot = Rotor3::from_quaternion_array(camera.rotation)
                        * Rotor3::from_rotation_xz(mx * std::f32::consts::FRAC_PI_4 * DT);
                    // let mut rot = Rotor3::from_quaternion_array(camera.rotation)
                    //     * (Rotor3::from_rotation_xz(
                    //         std::f32::consts::FRAC_PI_2
                    //             * if input.is_key_pressed(VirtualKeyCode::R) {
                    //                 1.0
                    //             } else {
                    //                 0.0
                    //             },
                    //     ));
                    rot.normalize();
                    camera.rotation = rot.into_quaternion_array();
                    let dx = input.key_axis(winit::event::VirtualKeyCode::A, winit::event::VirtualKeyCode::D);
                    let dz = input.key_axis(winit::event::VirtualKeyCode::W, winit::event::VirtualKeyCode::S);
                    let mut dir: ultraviolet::Vec3 = ultraviolet::Vec3 { x: (dx), y: (0.0), z: (dz) };


                    //println!("{}, {}, {}, {}", rot.into_quaternion_array()[0], rot.into_quaternion_array()[1], rot.into_quaternion_array()[2], rot.into_quaternion_array()[3],);
                    // if x or y are not 0
                    let here = if dir.mag_sq() > 0.0 {
                        dir.normalize();
                        ultraviolet::Vec3::from(camera.translation) + rot * dir * PLAYER_SPEED * DT
                    } else {
                        ultraviolet::Vec3::from(camera.translation)
                    };

                    //dbg!(rot.into_angle_plane().0);
                    //dbg!(dir, here);
                    //println!("{}", here);
                    camera.translation = here.into();
                    println!("{}, {}, {}", camera.translation[0], camera.translation[1], camera.translation[2]);
                    // frend.meshes.upload_meshes(&frend.gpu, fox_mesh, 0, ..);
                    //println!("tick");
                    //update_game();
                    // camera.screen_pos[0] += 0.01;
                    input.next_frame();

                    // MOVEMENT!
                    // arrow key or WASD movement
                    // player_transform.translation[2] goes UP when we walk forwards (yaw = 0)
                    /*

                    // TODO: make it so movement now deals with sin and cos to move in the right direction based on rotation

                    let mut current_yaw_degrees = fpcamera.yaw * 180.0 / PI;
                    if current_yaw_degrees < 0.0 {
                        current_yaw_degrees += 360.0;
                    }
                    let current_yaw_radians = current_yaw_degrees * (PI / 180.0);
                    if input.is_key_down(winit::event::VirtualKeyCode::Right)
                        || input.is_key_down(winit::event::VirtualKeyCode::D)
                    {
                        player_transform.translation[0] -=
                            PLAYER_SPEED * DT * f32::cos(current_yaw_radians);
                        player_transform.translation[2] +=
                            PLAYER_SPEED * DT * f32::sin(current_yaw_radians);
                    } else if input.is_key_down(winit::event::VirtualKeyCode::Left)
                        || input.is_key_down(winit::event::VirtualKeyCode::A)
                    {
                        player_transform.translation[0] +=
                            PLAYER_SPEED * DT * f32::cos(current_yaw_radians);
                        player_transform.translation[2] -=
                            PLAYER_SPEED * DT * f32::sin(current_yaw_radians);
                    }

                    if input.is_key_down(winit::event::VirtualKeyCode::Down)
                        || input.is_key_down(winit::event::VirtualKeyCode::S)
                    {
                        player_transform.translation[0] +=
                            PLAYER_SPEED * DT * f32::sin(current_yaw_radians);
                        player_transform.translation[2] -=
                            PLAYER_SPEED * DT * f32::cos(current_yaw_radians);
                    } else if input.is_key_down(winit::event::VirtualKeyCode::Up)
                        || input.is_key_down(winit::event::VirtualKeyCode::W)
                    {
                        player_transform.translation[0] -=
                            PLAYER_SPEED * DT * f32::sin(current_yaw_radians);
                        player_transform.translation[2] +=
                            PLAYER_SPEED * DT * f32::cos(current_yaw_radians);
                    }

                    // shortcut for resetting camera rotation
                    if input.is_key_down(winit::event::VirtualKeyCode::R) {
                        println!("resetting camera rotation...");
                        fpcamera.pitch = 0.0;
                        fpcamera.yaw = 0.0;
                    }

                    // shortcut for resetting camera position
                    if input.is_key_down(winit::event::VirtualKeyCode::T) {
                        println!("resetting camera position...");
                        player_transform.translation = Vec3 {
                            x: 0.0,
                            y: 0.0,
                            z: 0.0,
                        }
                        .into();
                    }

                    println!(
                        "sin: {}, cos: {}, pos x: {}, pos z: {}",
                        f32::sin(current_yaw_radians),
                        f32::cos(current_yaw_radians),
                        player_transform.translation[0],
                        player_transform.translation[2]
                    );
                    //println!("{}, {}", player_transform.translation[0], player_transform.translation[2]);*/
                }
                // Render prep

                // "update" updates the fpcamera's pitch and yaw, also sets fpcamera's position and rotation to the player_transform's position and rotation
                //fpcamera.update(&input, &player_transform);
                // "update_camera" sets the actual camera's translation and rotation to fpcamera's
                //fpcamera.update_camera(&mut camera);
                //player_transform.rotation = fpcamera.player_rot.into();
                frend.meshes.set_camera(&frend.gpu, camera);
                frend.flats.set_camera(&frend.gpu, camera);
                // update sprite positions and sheet regions
                // ok now render.
                
                

                //frend.render();
                // THIS LINE CAN BE REPLACED BY the following lines:
                let (frame, view, mut encoder) = frend.render_setup();
                {
                    // This is us manually making a renderpass
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: frenderer::wgpu::Operations {
                                load: wgpu::LoadOp::Clear(BACKGROUND_COLOR),
                                store: true,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &frend.gpu.depth_texture_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: true,
                            }),
                            stencil_ops: None,
                        }),
                    });
                    // frend has render_into to do the actual rendering
                    frend.render_into(&mut rpass);
                }
                // // This just submits the command encoder and presents the frame, we wouldn't need it if we did that some other way.
                frend.render_finish(frame, encoder);

                window.request_redraw();
            }
            event => {
                if frend.process_window_event(&event) {
                    window.request_redraw();
                }
                input.process_input_event(&event);
            }
        }
    });
}
