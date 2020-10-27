mod nnet;
mod three_sim;
use nnet::NeuralNetwork;
use queues::*;
use std::time::Instant;
use std::{
    io::{stdin, BufRead},
    path::Path,
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
) {
    let mut sample_iter = sample.get_iter();
    let mut avg_cost = AvgCost::new(100);
    let mut prev_position_guess = [0.0, 0.0, 0.0];
    let mut prev_rotation_guess = (0.0, [1.0, 0.0, 0.0]);
    let mut nnet_gradbuf = nnet.gen_gradbuf();
    if passes == 0 {
        let mut simulation = Simulation::new(model_path);
        let mut pass = 1;
        let mut last_pass_num: u32 = 0;
        'main_loop: loop {
            let (position, rotation) = sample_iter.next().unwrap();
            let points = three_sim::transform_points(&simulation.points, position, rotation);
            let (position_guess, rotation_guess, cost) =
                nnet.train(&points, position, rotation, &mut nnet_gradbuf, gradcoeff);
            if pass % packet_size == 0 {
                nnet.apply_gradbuf(&nnet_gradbuf, pass - last_pass_num);
                last_pass_num = pass;
            }
            println!(
                "Pass {:^8} avg.cost: {:.10} cost: {:.10}",
                pass,
                avg_cost.add(cost),
                cost
            );
            if !visualize_guess(
                &mut simulation,
                position,
                rotation,
                prev_position_guess,
                prev_rotation_guess,
                position_guess,
                rotation_guess,
                pause,
            ) {
                return;
            };
            prev_position_guess = position_guess;
            prev_rotation_guess = rotation_guess;
            pass += 1;
            let mut wt = 0;
            while simulation.handle() {
                if wt >= pause {
                    continue 'main_loop;
                }
                wt += 1;
            }
            return;
        }
    } else {
        let (points, _) = three_sim::load_model_points(model_path);
        let mut skip_prints = 0;
        let mut skip_prints_i = 0;
        const PRINT_PERIOD_MILLIS: u32 = 100;
        let mut last_pass_num: u32 = 0;
        for pass in 1..passes + 1 {
            let now = Instant::now();
            let (position, rotation) = sample_iter.next().unwrap();
            let points = three_sim::transform_points(&points, position, rotation);
            let (_, _, cost) =
                nnet.train(&points, position, rotation, &mut nnet_gradbuf, gradcoeff);
            if pass % packet_size == 0 || pass == passes {
                nnet.apply_gradbuf(&nnet_gradbuf, pass - last_pass_num);
                last_pass_num = pass;
            }
            if skip_prints_i >= skip_prints || pass == passes {
                println!(
                    "Pass {:^8}/{:^8} ({:^8.3}%) avg.cost: {:.10} cost: {:.10}",
                    pass,
                    passes,
                    pass as f64 / passes as f64 * 100.0,
                    avg_cost.add(cost),
                    cost
                );
                skip_prints_i = 0;
            } else {
                skip_prints_i += 1;
            }
            skip_prints = PRINT_PERIOD_MILLIS / now.elapsed().as_millis().max(1) as u32;
        }
    }
}

fn nnet_guess(model_path: &str, nnet: &NeuralNetwork, pause: u32) {
    let mut simulation = Simulation::new(model_path);
    let mut prev_position_guess = [0.0, 0.0, 0.0];
    let mut prev_rotation_guess = (0.0, [1.0, 0.0, 0.0]);
    'main_loop: loop {
        let (position, rotation) = nnet::generate_random();
        let points = three_sim::transform_points(&simulation.points, position, rotation);
        let (position_guess, rotation_guess) = nnet.guess(&points);
        let cost = NeuralNetwork::calc_cost(position, rotation, position_guess, rotation_guess);
        println!("Cost: {}", cost);
        if !visualize_guess(
            &mut simulation,
            position,
            rotation,
            prev_position_guess,
            prev_rotation_guess,
            position_guess,
            rotation_guess,
            pause,
        ) {
            return;
        };
        prev_position_guess = position_guess;
        prev_rotation_guess = rotation_guess;
        let mut wt = 0;
        while simulation.handle() {
            if wt >= pause {
                continue 'main_loop;
            }
            wt += 1;
        }
        return;
    }
}

fn demo(model_path: &str, pause: u32, sample: Option<&three_sim::Sample>) {
    let mut simulation = Simulation::new(model_path);
    let mut prev_position = [0.0, 0.0, 0.0];
    let mut prev_rotation = (0.0, [1.0, 0.0, 0.0]);
    let empty_sample = three_sim::Sample::new_empty();
    let mut sample_iter = if sample.is_none() {
        empty_sample.get_iter()
    } else {
        sample.unwrap().get_iter()
    };
    'main_loop: loop {
        let (position, rotation) = if sample.is_none() {
            nnet::generate_random()
        } else {
            sample_iter.next().unwrap()
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
        ) {
            return;
        };
        prev_position = position;
        prev_rotation = rotation;
        let mut wt = 0;
        let wt_steps = 20;
        while simulation.handle() {
            if wt >= wt_steps {
                continue 'main_loop;
            }
            wt += 1;
        }
        return;
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

    while simulation.handle() {
        let (model_position, model_rotation) = three_sim::transform_time(
            &prev_position_guess,
            prev_rotation_guess,
            &position_guess,
            rotation_guess,
            t as f64 / pause as f64,
        );

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

        if t >= pause {
            return true;
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
                println!("No nnet in model. Do you want to create one? (y/n)");
                if readline() == "y" {
                    println!("Creating nnet...");
                    let (model_points, _) = three_sim::load_model_points(&model_path);
                    if let Some(geometry) = pick_geometry(model_points.len() as u32) {
                        let nnet = nnet::NeuralNetwork::new(geometry);
                        println!("Nnet created, saving model...");
                        three_sim::write_model_nnet(&model_path, nnet::save_nnet(&nnet));
                        println!("Model saved.");
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

fn delete_nnet(model_path: &str) {
    println!(
        "Are you really want to delete nnet from \"{}\"? (y/n)",
        model_path
    );
    if readline() == "y" {
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

fn pick_demo(model_path: &str, pause: u32, sample: &three_sim::Sample) {
    println!("Use sample? (y/n)");
    if readline() == "y" {
        demo(model_path, pause, Some(sample));
    } else {
        demo(model_path, pause, None);
    }
}

fn save_model_nnet(model_path: &str, nnet: &NeuralNetwork) -> bool {
    println!("Save model \"{}\"? (y/n)", model_path);
    if readline() == "y" {
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
                set pac(k)et size
                set gradcoef(f)
                set (p)ause
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
                            pause,
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
                            pause,
                        );
                        changed = true;
                    }
                    "g" => nnet_guess(&model_path, &mut nnet, pause),
                    "c" => {
                        if let Some((new_model_path, new_nnet)) = pick_model_nnet() {
                            model_path = new_model_path;
                            nnet = new_nnet;
                        }
                    }
                    "s" => changed = !save_model_nnet(&model_path, &nnet),
                    "d" => delete_nnet(&model_path),
                    "t" => {
                        if let Some(s) = pick_trajectory() {
                            sample = s;
                        }
                    }
                    "k" => packet_size = pick_packet_size(packet_size),
                    "f" => gradcoeff = pick_gradcoeff(gradcoeff),
                    "p" => pause = pick_pause(pause),
                    "m" => pick_demo(&model_path, pause, &sample),
                    "e" => {
                        if changed {
                            save_model_nnet(&model_path, &nnet);
                        }
                        return;
                    }
                    _ => {}
                }
            }
        }
    }
}
