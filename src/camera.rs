use frenderer::{Camera3D, Transform3D};
use glam::*;

pub struct Camera {
    pub pitch: f32,
    pub yaw: f32,
    pub player_pos: Vec3,
    pub player_rot: Quat,
}
impl Camera {
    pub fn new() -> Self {
        Self {
            pitch: 0.0,
            yaw: 0.0,
            player_pos: Vec3::ZERO,
            player_rot: Quat::default(),
        }
    }
    pub fn update(&mut self, input: &frenderer::input::Input, player: &Transform3D) {
        use frenderer::input::MousePos;
        use std::f32::consts::FRAC_PI_2;
        let MousePos { y: dy, x: dx } = input.mouse_delta();
        self.pitch -= super::DT as f32 * dy as f32 / 10.0;
        self.yaw -= super::DT as f32 * dx as f32 / 10.0;
        // Make sure pitch isn't directly up or down (that would put
        // `eye` and `at` at the same z, which is Bad)
        self.pitch = self.pitch.clamp(-FRAC_PI_2 + 0.001, FRAC_PI_2 - 0.001);
        self.player_pos = player.translation.into();
        self.player_rot = Quat::from_array(player.rotation);
    }
    pub fn update_camera(&self, c: &mut Camera3D) {
        // The camera's position is offset from the player's position.
        c.translation = (self.player_pos
        // So, <0, 25, 2> in the player's local frame will need
        // to be rotated into world coordinates. Multiply by the player's rotation:
            + self.player_rot * Vec3::new(0.0, 25.0, 2.0)).into();

        // Next is the trickiest part of the code.
        // We want to rotate the camera around the way the player is
        // facing, then rotate it more to pitch is up or down.

        // We need to turn this rotation into a target vector (at) by
        // picking a point a bit "in front of" the eye point with
        // respect to our rotation.  This means composing two
        // rotations (player and camera) and rotating the unit forward
        // vector around by that composed rotation, then adding that
        // to the camera's position to get the target point.
        // So, we're adding a position and an offset to obtain a new position.

        c.rotation = (self.player_rot * (Quat::from_rotation_x(self.pitch) * Quat::from_rotation_y(self.yaw))).into();

    }
}