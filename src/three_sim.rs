use cgmath::Quaternion;
use std::{
    fs::File,
    io::{self, BufRead, Write},
    time::Instant,
};
use three::{camera::Camera, controls, material, Geometry, Group, Object, Window};

pub struct Simulation {
    window: Window,
    last_frame: Instant,
    deltaframe_ms: u32,
    camera: Camera,
    controls: controls::Orbit,
    pub model_group: Group,
    pub points_group: Group,
    pub points: Vec<[f64; 3]>,
}

/*pub fn _points_mean(points: &Vec<[f64; 3]>) -> [f64; 3] {
    let n = points.len();
    points.iter().fold([0.0, 0.0, 0.0], |acc, point| {
        [
            acc[0] + point[0] / n as f64,
            acc[1] + point[1] / n as f64,
            acc[2] + point[2] / n as f64,
        ]
    })
}*/

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

    pub fn new_empty() -> Sample {
        Sample(vec![])
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

const MODEL_SCALE_FACTOR: f64 = 0.01;

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
        /*meshes[0].set_material(material::Basic {
            color: 0x00ff00,
            map: None,
        });*/
        //meshes[0].set_material(material::Wireframe { color: 0x00ff00 });
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
        let camera = window.factory.perspective_camera(60.0, 1.0..1000.0);
        let controls = controls::Orbit::builder(&camera)
            .position([0.0, 0.0, 5.0])
            .target([0.0, 0.0, 0.0])
            .up([0.0, 1.0, 0.0])
            .build();

        let hemi_light = window.factory.hemisphere_light(0xffffbb, 0x080802, 1.0);
        hemi_light.look_at([15.0, 35.0, 35.0], [0.0, 0.0, 2.0], None);
        window.scene.add(&hemi_light);

        let grid_group = window.factory.group();
        for i in -20..21 {
            let grid_z = {
                let geometry = three::Geometry::plane(4.0, 0.001);
                let material = three::material::Wireframe { color: 0xc8bfe7 };
                window.factory.mesh(geometry, material)
            };
            grid_z.set_position([0.0, i as f32 / 10.0, 0.0]);
            grid_group.add(&grid_z);

            let grid_y = {
                let geometry = three::Geometry::plane(0.001, 4.0);
                let material = three::material::Wireframe { color: 0xc8bfe7 };
                window.factory.mesh(geometry, material)
            };
            grid_y.set_position([i as f32 / 10.0, 0.0, 0.0]);
            grid_group.add(&grid_y);
        }
        window.scene.add(&grid_group);

        let root = window.factory.group();
        window.scene.add(&root);

        let (model_group, points_group, points) =
            Simulation::load_model(&mut window, &root, model_path);

        Simulation {
            window,
            last_frame: Instant::now(),
            deltaframe_ms: 0,
            camera,
            controls,
            model_group,
            points_group,
            points,
        }
    }

    pub fn handle(&mut self) -> bool {
        if self.last_frame.elapsed().as_millis() as u32 > self.deltaframe_ms {
            self.last_frame = Instant::now();

            self.controls.update(&self.window.input);
            if self.window.input.hit(three::Key::Equals) {
                self.deltaframe_ms += 1;
            } else if self.window.input.hit(three::Key::Minus) {
                if self.deltaframe_ms > 0 {
                    self.deltaframe_ms -= 1;
                }
            } else if self.window.input.hit(three::Key::Key0) {
                self.deltaframe_ms = 0;
            }
            self.window.render(&self.camera);
            if !self.window.update() {
                return false;
            }
        }
        !self.window.input.hit(three::KEY_ESCAPE)
    }
}
