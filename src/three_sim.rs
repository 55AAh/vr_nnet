use cgmath::Quaternion;
use std::{
    fs::File,
    io::{self, BufRead, Write},
    time::Instant,
};
use three::{camera::Camera, material, Geometry, Group, Object, Text, Window};

pub struct Sample(Vec<([f64; 3], (f64, [f64; 3]))>);

impl Sample {
    pub fn new(sample_path: &str) -> Option<Sample> {
        let mut samples = Vec::<([f64; 3], (f64, [f64; 3]))>::new();
        for line in io::BufReader::new(File::open(sample_path).unwrap())
            .lines()
            .into_iter()
            .map(|line| line.unwrap_or("".to_string()))
        {
            let line_split = line.split(",").collect::<Vec<&str>>();
            if line_split.len() != 7 {
                print!("Bad sample: \"{}\"!", line);
                return None;
            }
            let mut line_nums = Vec::<f64>::new();
            for ln in line_split {
                if let Ok(n) = ln.parse::<f64>() {
                    line_nums.push(n);
                } else {
                    print!("Invalid number: \"{}\"!", ln);
                    return None;
                }
            }
            samples.push((
                [line_nums[0], line_nums[1], line_nums[2]],
                (line_nums[3], [line_nums[4], line_nums[5], line_nums[6]]),
            ))
        }
        Some(Sample(samples))
    }

    pub fn get_iter(
        &self,
    ) -> std::iter::Cycle<std::iter::Cloned<std::slice::Iter<([f64; 3], (f64, [f64; 3]))>>> {
        self.0.iter().cloned().cycle()
    }
}

pub fn transform_time(
    prev_shift: &[f64; 3],
    prev_rotation: (f64, [f64; 3]),
    shift: &[f64; 3],
    rotation: (f64, [f64; 3]),
    t: f64,
) -> ([f64; 3], (f64, [f64; 3])) {
    let q = Quaternion::new(
        prev_rotation.0,
        prev_rotation.1[0],
        prev_rotation.1[1],
        prev_rotation.1[2],
    )
    .slerp(
        Quaternion::new(rotation.0, rotation.1[0], rotation.1[1], rotation.1[2]),
        t,
    );
    (
        [
            prev_shift[0] * (1.0 - t) + shift[0] * t,
            prev_shift[1] * (1.0 - t) + shift[1] * t,
            prev_shift[2] * (1.0 - t) + shift[2] * t,
        ],
        (q.s, [q.v[0], q.v[1], q.v[2]]),
    )
}

pub fn transform_points(
    points: &Vec<[f64; 3]>,
    shift: [f64; 3],
    rotation: (f64, [f64; 3]),
) -> Vec<[f64; 3]> {
    points
        .iter()
        .map(|point| quaternion::rotate_vector(rotation, *point))
        .map(|point| {
            [
                point[0] + shift[0],
                point[1] + shift[1],
                point[2] + shift[2],
            ]
        })
        .collect()
}

const MODEL_SCALE_FACTOR: f64 = 0.05;

pub fn load_model_points(model_path: &str) -> (Vec<[f64; 3]>, [f64; 3]) {
    let (mut models, _) = tobj::load_obj(model_path, false).unwrap();
    let mut mean = [0.0, 0.0, 0.0];
    let mut vertices = Vec::new();
    for vertex in models.drain(..).next().unwrap().mesh.positions.chunks(3) {
        let vertex = [
            vertex[0] as f64 * MODEL_SCALE_FACTOR,
            vertex[1] as f64 * MODEL_SCALE_FACTOR,
            vertex[2] as f64 * MODEL_SCALE_FACTOR,
        ];
        mean[0] += vertex[0];
        mean[1] += vertex[1];
        mean[2] += vertex[2];
        vertices.push(vertex);
    }
    let n = vertices.len() as f64;
    let mean = [mean[0] / n, mean[1] / n, mean[2] / n];

    let mut points = Vec::new();
    for vertex in vertices {
        let position = [
            vertex[0] - mean[0],
            vertex[1] - mean[1],
            vertex[2] - mean[2],
        ];
        points.push(position);
    }

    (points, mean)
}

pub fn get_model_nnet(model_path: &str) -> Vec<String> {
    let lines_iter = io::BufReader::new(File::open(model_path).unwrap())
        .lines()
        .into_iter()
        .map(|line| line.unwrap_or("".to_string()));
    let mut lines = Vec::<String>::new();
    let mut found = false;
    for line in lines_iter {
        if line == "# NNET" {
            found = true;
        }
        if found {
            if line.chars().next() != Some('#') {
                return vec![];
            }
            lines.push(line.chars().skip(1).collect());
        }
    }
    lines
}

pub fn write_model_nnet(model_path: &str, lines: Vec<String>) {
    let mut total_lines = Vec::<String>::new();
    for line in io::BufReader::new(File::open(model_path).unwrap())
        .lines()
        .map(|line| line.unwrap_or("".to_string()))
    {
        if line == "# NNET" {
            break;
        } else {
            total_lines.push(line);
        }
    }
    total_lines.extend(lines.iter().map(|line| "#".to_string() + line));

    io::BufWriter::new(File::create(model_path).unwrap())
        .write_all(total_lines.join("\n").as_bytes())
        .unwrap();
}

pub struct Simulation {
    window: Window,
    pos_text: Text,
    pub pos_text_visible: bool,
    pos_text_visible_b: bool,
    last_frame: Instant,
    deltaframe_ms: u32,
    camera: Camera,
    camera_zoom: f32,
    camera_rot: f32,
    wheel_pos: f32,
    wheel_mult: f32,
    grid: Group,
    grid_visible: bool,
    grid_visible_b: bool,
    pub model_group: Group,
    pub points_group: Group,
    pub points: Vec<[f64; 3]>,
}

impl Simulation {
    fn load_model(
        window: &mut Window,
        root: &Group,
        model_path: &str,
    ) -> (Group, Group, Vec<[f64; 3]>) {
        let model_group = window.factory.group();
        let (mut group_maps, meshes) = window.factory.load_obj(model_path);
        if group_maps.len() != 1 || meshes.len() != 1 {
            panic!("Model should have one group!");
        }
        let (_, group_map) = group_maps.drain().next().unwrap();
        group_map.set_scale(MODEL_SCALE_FACTOR as f32);
        model_group.add(&group_map);

        let points_group = window.factory.group();
        let (points, mean) = load_model_points(model_path);
        for position in &points {
            let geometry = Geometry::uv_sphere(0.01, 5, 5);
            let material = material::Basic {
                color: 0xff0000,
                map: None,
            };
            let point = window.factory.mesh(geometry, material);
            point.set_position([
                (position[0] + mean[0]) as f32,
                (position[1] + mean[1]) as f32,
                (position[2] + mean[2]) as f32,
            ]);

            points_group.add(&point);
        }

        root.add(&model_group);
        root.add(&points_group);

        (model_group, points_group, points)
    }

    pub fn new(model_path: &str) -> Self {
        let mut window = Window::new("vr_nnet");
        let camera = window.factory.perspective_camera(60.0, 0.0001..1000.0);

        let font = window.factory.load_font_karla();
        let pos_text = window.factory.ui_text(&font, "");

        let (mut group_maps, meshes) = window.factory.load_obj("Plane.obj");
        let (_, grid) = group_maps.drain().next().unwrap();
        meshes[0].set_material(material::Wireframe { color: 0xffffff });
        grid.set_scale(0.1);
        window.scene.add(&grid);

        let hemi_light = window.factory.hemisphere_light(0xffffbb, 0x080802, 1.0);
        hemi_light.look_at([15.0, 35.0, 35.0], [0.0, 0.0, 2.0], None);
        window.scene.add(&hemi_light);

        let root = window.factory.group();
        window.scene.add(&root);

        let (model_group, points_group, points) =
            Simulation::load_model(&mut window, &root, model_path);

        let mut simulation = Simulation {
            window,
            pos_text,
            pos_text_visible: false,
            pos_text_visible_b: false,
            last_frame: Instant::now(),
            deltaframe_ms: 70,
            camera,
            camera_rot: 0.0,
            camera_zoom: 5.0,
            wheel_pos: 0.0,
            wheel_mult: 1.0,
            grid,
            grid_visible: true,
            grid_visible_b: false,
            model_group,
            points_group,
            points,
        };
        simulation.update_camera();
        simulation
    }

    fn update_camera(&mut self) {
        self.camera.look_at(
            [
                self.camera_rot.sin() * self.camera_zoom,
                self.camera_rot.cos() * self.camera_zoom,
                3.0,
            ],
            [0.0, 0.0, 0.0],
            None,
        );
    }

    pub fn mouse_movement(&self) -> [f32; 2] {
        self.window
            .input
            .mouse_movements()
            .iter()
            .fold([0.0, 0.0], |acc, new| [acc[0] + new.x, acc[0] + new.y])
    }

    pub fn wheel_movement(&self) -> f32 {
        self.window
            .input
            .mouse_wheel_movements()
            .iter()
            .sum::<f32>()
    }

    pub fn update_mode(&self, mode: &mut char) {
        if self.window.input.hit(three::Key::R) {
            *mode = 'r';
        }
        if self.window.input.hit(three::Key::S) {
            *mode = 's';
        }
        if self.window.input.hit(three::Key::M) {
            *mode = 'm';
        }
        if self.window.input.hit(three::Key::C) {
            *mode = 'c';
        }
    }

    pub fn mouse_position(&self) -> [f32; 2] {
        [
            (self.window.input.mouse_pos().x / self.window.size().x * 2.0 - 1.0) * self.camera_zoom,
            (self.window.input.mouse_pos().y / self.window.size().y * 2.0 - 1.0) * self.camera_zoom,
        ]
    }

    pub fn get_model_position(&mut self) -> ([f64; 3], (f64, [f64; 3])) {
        if !self
            .window
            .input
            .hit(three::Button::Mouse(three::MouseButton::Left))
        {
            self.wheel_pos +=
                self.wheel_movement() * self.window.input.delta_time() / 100000.0 * self.wheel_mult;
        }
        (
            [
                -self.mouse_position()[0] as f64,
                self.mouse_position()[1] as f64,
                self.wheel_pos as f64,
            ],
            (0.0, [0.0, 0.0, 1.0]),
        )
    }

    pub fn set_pos_text(&mut self, text: &str) {
        self.pos_text.set_text(text);
    }

    pub fn handle(&mut self, move_camera: bool) -> bool {
        if self.last_frame.elapsed().as_millis() as u32 > self.deltaframe_ms {
            self.last_frame = Instant::now();

            if self.window.input.hit(three::Key::T) {
                if !self.pos_text_visible_b {
                    self.pos_text_visible = !self.pos_text_visible;
                }
                self.pos_text_visible_b = true;
            } else {
                self.pos_text_visible_b = false;
            }

            if self.window.input.hit(three::Key::G) {
                if !self.grid_visible_b {
                    self.grid_visible = !self.grid_visible;
                    self.grid.set_visible(self.grid_visible);
                }
                self.grid_visible_b = true;
            } else {
                self.grid_visible_b = false;
            }

            if move_camera {
                if self
                    .window
                    .input
                    .hit(three::Button::Mouse(three::MouseButton::Left))
                {
                    self.camera_rot += self.mouse_movement()[0] / 300.0;
                    self.camera_zoom -= self.wheel_movement() / 100.0;
                    self.camera_zoom = self.camera_zoom.max(1.0).min(10.0);
                }
                self.update_camera();
            }

            if self.window.input.hit(three::Key::PageUp) {
                self.wheel_mult *= 1.1;
            } else if self.window.input.hit(three::Key::PageDown) {
                self.wheel_mult /= 1.1;
            }
            if self.window.input.hit(three::Key::PageUp)
                && self.window.input.hit(three::Key::PageDown)
            {
                self.wheel_mult = 1.0;
            }

            if self.window.input.hit(three::Key::Equals) {
                self.deltaframe_ms += 1;
            } else if self.window.input.hit(three::Key::Minus) {
                if self.deltaframe_ms > 0 {
                    self.deltaframe_ms -= 1;
                }
            } else if self.window.input.hit(three::Key::Key0) {
                self.deltaframe_ms = 70;
            }
            self.window.render(&self.camera);
            if !self.window.update() {
                return false;
            }
        }
        !self.window.input.hit(three::KEY_ESCAPE)
    }
}
