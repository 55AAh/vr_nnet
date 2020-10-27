mod nnet;
mod three_sim;
use nnet::NeuralNetwork;
use queues::*;
use std::{
    fs::File,
    io,
    io::Write,
    io::{stdin, BufRead},
    path::Path,
    time::Instant,
};
use three::Object;
use three_sim::Simulation;

struct AvgCost {
    costs_queue: Queue<f64>,
    costs_queue_size: usize,
    costs_sum: f64,
}

impl AvgCost {
    pub fn new(costs_quese_size: usize) -> Self {
        Self {
            costs_queue: queue![],
            costs_queue_size: costs_quese_size,
            costs_sum: 0.0,
        }
    }

    pub fn add(&mut self, new_cost: f64) -> f64 {
        self.costs_queue.add(new_cost).unwrap();
        if self.costs_queue.size() > self.costs_queue_size {
            self.costs_sum -= self.costs_queue.remove().unwrap();
        }
        self.costs_sum += new_cost;
        self.costs_sum / self.costs_queue.size() as f64
    }
}

fn nnet_train(
    model_path: &str,
    nnet: &mut NeuralNetwork,
    sample: &three_sim::Sample,
    packet_size: u32,
    passes: u32,
    gradcoeff: f64,
    pause: u32,
    log: bool,
) {
    let mut avg_cost = AvgCost::new(100);
    let mut prev_position_guess = [0.0, 0.0, 0.0];
    let mut prev_rotation_guess = (0.0, [0.0, 0.0, 1.0]);
    let mut nnet_gradbuf = nnet.gen_gradbuf();
    let mut skip_prints = 0;
    let mut skip_prints_i = 0;
    let mut sample_iter = sample.get_iter();
    let mut mode: char = 'c';
    const PRINT_PERIOD_MILLIS: u32 = 1000;
    let mut buf_writer = io::BufWriter::new(if log {
        File::create("passes_log.txt").unwrap()
    } else {
        if let Ok(i) = File::open("passes_log.txt") {
            i
        } else {
            File::create("passes_log.txt").unwrap()
        }
    });
    if passes == 0 {
        let mut simulation = Simulation::new(model_path);
        let mut pass = 1;
        let mut last_pass_num: u32 = 0;
        loop {
            let now = Instant::now();
            let (position, rotation) = match mode {
                'r' => nnet::generate_random(),
                's' => sample_iter.next().unwrap(),
                'm' | 'c' => simulation.get_model_position(),
                _ => ([0.0, 0.0, 0.0], (0.0, [0.0, 0.0, 1.0])),
            };
            let points = three_sim::transform_points(&simulation.points, position, rotation);
            let (position_guess, rotation_guess) = if mode != 'c' {
                let (position_guess, rotation_guess, cost) =
                    nnet.train(&points, position, rotation, &mut nnet_gradbuf, gradcoeff);
                if pass % packet_size == 0 {
                    nnet.apply_gradbuf(&mut nnet_gradbuf, pass - last_pass_num);
                    last_pass_num = pass;
                }
                let mut print_str = format!(
                    "Pass {:^8} avg.cost: {:.10} cost: {:.10}",
                    pass,
                    avg_cost.add(cost),
                    cost
                );
                pass += 1;
                if skip_prints_i >= skip_prints || pass == passes {
                    println!("{}", print_str);
                    skip_prints_i = 0;
                } else {
                    skip_prints_i += 1;
                }
                if log {
                    print_str += "\n";
                    buf_writer.write(print_str.as_bytes()).unwrap();
                }
                skip_prints = PRINT_PERIOD_MILLIS / now.elapsed().as_millis().max(1) as u32;
                (position_guess, rotation_guess)
            } else {
                (position, rotation)
            };
            if !visualize_guess(
                &mut simulation,
                position,
                rotation,
                prev_position_guess,
                prev_rotation_guess,
                position_guess,
                rotation_guess,
                pause,
                &mut mode,
            ) {
                return;
            }
            let (new_prev_position_guess, new_prev_rotation_guess) = if mode == 'm' {
                three_sim::transform_time(
                    &prev_position_guess,
                    prev_rotation_guess,
                    &position_guess,
                    rotation_guess,
                    1.0 / pause as f64,
                )
            } else {
                (position_guess, rotation_guess)
            };
            prev_position_guess = new_prev_position_guess;
            prev_rotation_guess = new_prev_rotation_guess;
        }
    } else {
        mode = if pick_yn("Use sample?") { 's' } else { 'r' };
        let (points, _) = three_sim::load_model_points(model_path);
        let mut last_pass_num: u32 = 0;
        for pass in 1..passes + 1 {
            let now = Instant::now();
            let (position, rotation) = match mode {
                'r' => nnet::generate_random(),
                's' => sample_iter.next().unwrap(),
                _ => ([0.0, 0.0, 0.0], (0.0, [0.0, 0.0, 1.0])),
            };
            let points = three_sim::transform_points(&points, position, rotation);
            let (_, _, cost) =
                nnet.train(&points, position, rotation, &mut nnet_gradbuf, gradcoeff);
            if pass % packet_size == 0 || pass == passes {
                nnet.apply_gradbuf(&mut nnet_gradbuf, pass - last_pass_num);
                last_pass_num = pass;
            }
            let mut print_str = format!(
                "Pass {:^8}/{:^8} ({:^8.3}%) avg.cost: {:.10} cost: {:.10}",
                pass,
                passes,
                pass as f64 / passes as f64 * 100.0,
                avg_cost.add(cost),
                cost
            );
            if skip_prints_i >= skip_prints || pass == passes {
                println!("{}", print_str);
                skip_prints_i = 0;
            } else {
                skip_prints_i += 1;
            }
            skip_prints = PRINT_PERIOD_MILLIS / now.elapsed().as_millis().max(1) as u32;
            if log {
                print_str += "\n";
                buf_writer.write(print_str.as_bytes()).unwrap();
            }
        }
    }
}

fn nnet_guess(model_path: &str, nnet: &NeuralNetwork, sample: &three_sim::Sample, pause: u32) {
    let mut simulation = Simulation::new(model_path);
    let mut prev_position_guess = [0.0, 0.0, 0.0];
    let mut prev_rotation_guess = (0.0, [0.0, 0.0, 1.0]);
    let mut sample_iter = sample.get_iter();
    let mut mode: char = 'c';
    loop {
        let (position, rotation) = match mode {
            'r' => nnet::generate_random(),
            's' => sample_iter.next().unwrap(),
            'm' | 'c' => simulation.get_model_position(),
            _ => ([0.0, 0.0, 0.0], (0.0, [0.0, 0.0, 1.0])),
        };
        let points = three_sim::transform_points(&simulation.points, position, rotation);
        let (position_guess, rotation_guess) = if mode != 'c' {
            let (position_guess, rotation_guess) = nnet.guess(&points);
            let cost = NeuralNetwork::calc_cost(position, rotation, position_guess, rotation_guess);
            println!("Cost: {}", cost);
            (position_guess, rotation_guess)
        } else {
            (position, rotation)
        };
        if !visualize_guess(
            &mut simulation,
            position,
            rotation,
            prev_position_guess,
            prev_rotation_guess,
            position_guess,
            rotation_guess,
            pause,
            &mut mode,
        ) {
            return;
        };
        let (new_prev_position_guess, new_prev_rotation_guess) = if mode == 'm' {
            three_sim::transform_time(
                &prev_position_guess,
                prev_rotation_guess,
                &position_guess,
                rotation_guess,
                1.0 / pause as f64,
            )
        } else {
            (position_guess, rotation_guess)
        };
        prev_position_guess = new_prev_position_guess;
        prev_rotation_guess = new_prev_rotation_guess;
    }
}

fn demo(model_path: &str, sample: &three_sim::Sample, pause: u32) {
    let mut simulation = Simulation::new(model_path);
    let mut prev_position = [0.0, 0.0, 0.0];
    let mut prev_rotation = (0.0, [0.0, 0.0, 1.0]);
    let mut sample_iter = sample.get_iter();
    let mut mode: char = 'c';
    loop {
        let (position, rotation) = match mode {
            'r' => nnet::generate_random(),
            's' => sample_iter.next().unwrap(),
            'm' | 'c' => simulation.get_model_position(),
            _ => ([0.0, 0.0, 0.0], (0.0, [0.0, 0.0, 1.0])),
        };
        if !visualize_guess(
            &mut simulation,
            position,
            rotation,
            prev_position,
            prev_rotation,
            position,
            rotation,
            pause,
            &mut mode,
        ) {
            return;
        };
        let (new_prev_position, new_prev_rotation) = if mode == 'm' {
            three_sim::transform_time(
                &prev_position,
                prev_rotation,
                &position,
                rotation,
                1.0 / (pause as f64),
            )
        } else {
            (position, rotation)
        };
        prev_position = new_prev_position;
        prev_rotation = new_prev_rotation;
    }
}

fn visualize_guess(
    simulation: &mut Simulation,
    position: [f64; 3],
    rotation: (f64, [f64; 3]),
    prev_position_guess: [f64; 3],
    prev_rotation_guess: (f64, [f64; 3]),
    position_guess: [f64; 3],
    rotation_guess: (f64, [f64; 3]),
    pause: u32,
    mode: &mut char,
) -> bool {
    simulation.points_group.set_transform(
        [position[0] as f32, position[1] as f32, position[2] as f32],
        [
            rotation.0 as f32,
            rotation.1[0] as f32,
            rotation.1[1] as f32,
            rotation.1[2] as f32,
        ],
        1.0,
    );

    let mut t = 0;

    while simulation.handle(*mode != 'm') {
        simulation.update_mode(mode);

        let (model_position, model_rotation) = three_sim::transform_time(
            &prev_position_guess,
            prev_rotation_guess,
            &position_guess,
            rotation_guess,
            t as f64 / pause as f64,
        );

        let pos_text = format!(
            "   Model\nX {}\nY {}\nZ {}\n\nW {}\nI {}\nJ {}\nK {}\n\n   Guess\nX {}\nY {}\nZ {}\n\nW {}\nI {}\nJ {}\nK {}",
            model_position[0],
            model_position[1],
            model_position[2],
            model_rotation.0,
            model_rotation.1[0],
            model_rotation.1[1],
            model_rotation.1[2],
            position_guess[0],
            position_guess[1],
            position_guess[2],
            rotation_guess.0,
            rotation_guess.1[0],
            rotation_guess.1[1],
            rotation_guess.1[2]
        );
        simulation.set_pos_text(if simulation.pos_text_visible {
            pos_text.as_str()
        } else {
            "*"
        });

        simulation.model_group.set_transform(
            [
                model_position[0] as f32,
                model_position[1] as f32,
                model_position[2] as f32,
            ],
            [
                model_rotation.0 as f32,
                model_rotation.1[0] as f32,
                model_rotation.1[1] as f32,
                model_rotation.1[2] as f32,
            ],
            1.0,
        );

        if t >= pause || *mode == 'm' {
            let mut wt = 0;
            while simulation.handle(*mode != 'm') {
                simulation.update_mode(mode);
                if wt >= pause || *mode == 'm' {
                    return true;
                }
                wt += 1;
            }
            return false;
        }
        t += 1;
    }
    return false;
}

fn pick_geometry(inputs_count: u32) -> Option<Vec<u32>> {
    let prompt = "Please provide the neurons count in intermediate layers (separated by spaces) or press enter to abort :";
    println!(
        "Nnet will have {} input and 7 outputs. {}",
        inputs_count, prompt
    );
    'main_loop: loop {
        let mut geometry = Vec::<u32>::new();
        geometry.push(inputs_count);
        let s = readline();
        if s == "" {
            return None;
        }
        for ns in s.split(" ") {
            if let Ok(n) = ns.parse::<u32>() {
                geometry.push(n);
            } else {
                println!("Invalid number: \"{}\"!\n{}", ns, prompt);
                continue 'main_loop;
            }
        }
        geometry.push(7);
        return Some(geometry);
    }
}

fn pick_model_nnet() -> Option<(String, NeuralNetwork)> {
    loop {
        println!("Model file name? (enter to abort):");
        let model_path = readline();
        if model_path == "" {
            return None;
        }
        if Path::new(&model_path).exists() {
            println!("Fetching nnet from model...");
            let lines = three_sim::get_model_nnet(&model_path);
            if lines.len() == 0 {
                if pick_yn("No nnet in model. Do you want to create one?") {
                    println!("Creating nnet...");
                    let (model_points, _) = three_sim::load_model_points(&model_path);
                    if let Some(geometry) = pick_geometry(model_points.len() as u32) {
                        let nnet = nnet::NeuralNetwork::new(geometry);
                        println!("Nnet created.");
                        return Some((model_path, nnet));
                    }
                }
            } else {
                if let Some(nnet) = nnet::read_nnet(lines) {
                    println!("Nnet fetched.");
                    return Some((model_path, nnet));
                } else {
                    println!("Unable to load nnet from model!");
                }
            }
        } else {
            println!("File doesn't exist!");
        }
    }
}

fn pick_packet_size(packet_size: u32) -> u32 {
    loop {
        println!("Packet size = {}, pick new packet size:", packet_size);
        let s = readline();
        if s == "" {
            return packet_size;
        }
        if let Ok(packet_size) = s.parse::<u32>() {
            if packet_size > 0 {
                return packet_size;
            }
        }
    }
}

fn pick_delete_nnet(model_path: &str) {
    if pick_yn(&format!(
        "Do you really want to delete nnet from \"{}\" and save model?",
        model_path
    )) {
        println!("Saving model...");
        three_sim::write_model_nnet(&model_path, vec![]);
        println!("Model saved.");
    }
}

fn pick_trajectory() -> Option<three_sim::Sample> {
    loop {
        println!("Sample file name? (enter to abort):");
        let sample_path = readline();
        if sample_path == "" {
            return None;
        }
        if Path::new(&sample_path).exists() {
            println!("Fetching sample...");
            if let Some(sample) = three_sim::Sample::new(&sample_path) {
                println!("Sample fetched.");
                return Some(sample);
            }
        } else {
            println!("File doesn't exist!");
        }
    }
}

fn pick_passes_count() -> u32 {
    loop {
        println!("Passes count?");
        if let Ok(passes) = readline().parse::<u32>() {
            if passes > 0 {
                return passes;
            }
        }
    }
}

fn pick_gradcoeff(gradcoeff: f64) -> f64 {
    loop {
        println!("Gradcoeff = {}, pick new value?", gradcoeff);
        let s = readline();
        if s == "" {
            return gradcoeff;
        }
        if let Ok(gradcoeff) = s.parse::<f64>() {
            if gradcoeff > 0.0 {
                return gradcoeff;
            }
        }
    }
}

fn pick_pause(pause: u32) -> u32 {
    loop {
        println!("Pause = {}, pick new pause?", pause);
        let s = readline();
        if s == "" {
            return pause;
        }
        if let Ok(pause) = s.parse::<u32>() {
            if pause > 0 {
                return pause;
            }
        }
    }
}

fn pick_save_model_nnet(model_path: &str, nnet: &NeuralNetwork) -> bool {
    if pick_yn(&format!("Save model \"{}\"?", model_path)) {
        println!("Saving model...");
        three_sim::write_model_nnet(model_path, nnet::save_nnet(nnet));
        println!("Model saved.");
        return true;
    }
    false
}

fn readline() -> String {
    stdin().lock().lines().next().unwrap().unwrap()
}

fn pick_yn(message: &str) -> bool {
    loop {
        println!("{} (y/n)", message);
        match readline().as_str() {
            "y" | "н" => return true,
            "n" | "т" => return false,
            _ => {}
        }
    }
}

fn main() {
    if let Some((mp, nn)) = pick_model_nnet() {
        if let Some(s) = pick_trajectory() {
            let mut model_path = mp;
            let mut nnet = nn;
            let mut sample = s;
            let mut packet_size: u32 = 100;
            let mut gradcoeff = 0.000001;
            let mut pause: u32 = 10;
            let mut changed = false;
            let mut log = true;
            loop {
                println!(
                    "
                (q)uick train
                (v)isual train
                (g)uess
                (c)hange model/nnet
                (s)ave model/nnet
                (d)elete nnet
                (t)rajectory file
                set de(l)taframe
                set pac(k)et size
                set gradcoef(f)
                set (p)ause
                set loggi(n)g
                (m)ove test
                (e)xit
                "
                );
                match readline().as_str() {
                    "q" => {
                        nnet_train(
                            &model_path,
                            &mut nnet,
                            &sample,
                            packet_size,
                            pick_passes_count(),
                            gradcoeff,
                            pause * 50,
                            log,
                        );
                        changed = true;
                    }
                    "v" => {
                        nnet_train(
                            &model_path,
                            &mut nnet,
                            &sample,
                            packet_size,
                            0,
                            gradcoeff,
                            pause * 50,
                            log,
                        );
                        changed = true;
                    }
                    "g" => nnet_guess(&model_path, &mut nnet, &sample, pause * 50),
                    "c" => {
                        if let Some((new_model_path, new_nnet)) = pick_model_nnet() {
                            model_path = new_model_path;
                            nnet = new_nnet;
                        }
                    }
                    "s" => changed = !pick_save_model_nnet(&model_path, &nnet),
                    "d" => pick_delete_nnet(&model_path),
                    "t" => {
                        if let Some(s) = pick_trajectory() {
                            sample = s;
                        }
                    }
                    "k" => packet_size = pick_packet_size(packet_size),
                    "f" => gradcoeff = pick_gradcoeff(gradcoeff),
                    "p" => pause = pick_pause(pause),
                    "n" => log = pick_yn("Enable passes logging?"),
                    "m" => demo(&model_path, &sample, pause * 50),
                    "e" => {
                        if changed {
                            pick_save_model_nnet(&model_path, &nnet);
                        }
                        return;
                    }
                    _ => {}
                }
            }
        }
    }
}
