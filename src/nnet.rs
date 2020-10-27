use std::f64::consts::PI;

use rand::Rng;

pub struct NeuralNetwork {
    c: Vec<Vec<Vec<f64>>>,
    b: Vec<Vec<f64>>,
}

impl NeuralNetwork {
    pub fn new(geometry: Vec<u32>) -> Self {
        let mut rng = rand::thread_rng();
        NeuralNetwork {
            c: (1..geometry.len())
                .map(|i| {
                    (0..geometry[i])
                        .map(|_| {
                            (0..if i == 1 {
                                geometry[0] * 3
                            } else {
                                geometry[i - 1]
                            })
                                .map(|_| rng.gen_range(-1.0, 1.0))
                                .collect()
                        })
                        .collect()
                })
                .collect(),
            b: (1..geometry.len())
                .map(|i| (0..geometry[i]).map(|_| rng.gen_range(-1.0, 1.0)).collect())
                .collect(),
        }
    }

    pub fn gen_gradbuf(&self) -> NeuralNetwork {
        let mut geometry: Vec<u32> = vec![self.c[0].len() as u32];
        geometry.extend(self.c.iter().map(|cl| cl.len() as u32));
        NeuralNetwork {
            c: (1..geometry.len())
                .map(|i| {
                    (0..geometry[i])
                        .map(|_| (0..geometry[i - 1]).map(|_| 0.0).collect())
                        .collect()
                })
                .collect(),
            b: (1..geometry.len())
                .map(|i| (0..geometry[i]).map(|_| 0.0).collect())
                .collect(),
        }
    }

    pub fn train(
        &mut self,
        points: &Vec<[f64; 3]>,
        position: [f64; 3],
        rotation: (f64, [f64; 3]),
        nnet_grad: &mut NeuralNetwork,
        gradcoeff: f64,
    ) -> ([f64; 3], (f64, [f64; 3]), f64) {
        let input = points.iter().flatten().cloned().collect::<Vec<f64>>();

        let mut vals = Vec::<Vec<f64>>::new();
        vals.push(input.clone());
        for i in 0..self.c.len() {
            vals.push(
                self.c[i]
                    .iter()
                    .zip(self.b[i].iter())
                    .map(|(c, b)| {
                        vals.last()
                            .unwrap()
                            .iter()
                            .zip(c.iter())
                            .map(|(val, coef)| val * coef)
                            .sum::<f64>()
                            + b
                    })
                    .collect::<Vec<f64>>(),
            );
        }

        let guess = vals.last().unwrap();
        let (position_guess, rotation_guess) = (
            [guess[0], guess[1], guess[2]],
            (guess[3], [guess[4], guess[5], guess[6]]),
        );
        let cost = NeuralNetwork::calc_cost(position, rotation, position_guess, rotation_guess);

        let correct = vec![
            position[0],
            position[1],
            position[2],
            rotation.0,
            rotation.1[0],
            rotation.1[1],
            rotation.1[2],
        ];
        let mut da = guess
            .iter()
            .zip(correct.iter())
            .map(|(g, c)| 2.0 * (g - c))
            .collect::<Vec<f64>>();

        for t in (0..self.c.len()).rev() {
            let da_p = if t > 0 {
                (0..self.c[t - 1].len())
                    .map(|k| {
                        vals[t + 1]
                            .iter()
                            .zip(&self.c[t])
                            .map(|(a, c)| a * c[k])
                            .sum()
                    })
                    .collect::<Vec<f64>>()
            } else {
                vec![]
            };

            for ((c, b), da) in nnet_grad.c[t]
                .iter_mut()
                .zip(nnet_grad.b[t].iter_mut())
                .zip(da.iter())
            {
                *b += da * gradcoeff;
                for (cx, a) in c
                    .iter_mut()
                    .zip(if t > 0 { &vals[t] } else { &input }.iter())
                {
                    *cx += da * a * gradcoeff;
                }
            }

            da = da_p;
        }

        (position_guess, rotation_guess, cost)
    }

    pub fn apply_gradbuf(&mut self, gradbuf: &NeuralNetwork, count: u32) {
        for t in 0..self.c.len() {
            for ((c, b), (gc, gb)) in self.c[t]
                .iter_mut()
                .zip(self.b[t].iter_mut())
                .zip(gradbuf.c[t].iter().zip(gradbuf.b[t].iter()))
            {
                *b -= *gb / count as f64;
                for (cx, gcx) in c.iter_mut().zip(gc.iter()) {
                    *cx -= *gcx / count as f64;
                }
            }
        }
    }

    pub fn guess(&self, points: &Vec<[f64; 3]>) -> ([f64; 3], (f64, [f64; 3])) {
        let mut vals = points.iter().flatten().cloned().collect::<Vec<f64>>();
        for i in 0..self.c.len() {
            vals = self.c[i]
                .iter()
                .zip(self.b[i].iter())
                .map(|(c, b)| {
                    vals.iter()
                        .zip(c.iter())
                        .map(|(val, coef)| val * coef)
                        .sum::<f64>()
                        + b
                })
                .collect::<Vec<f64>>();
        }
        (
            [vals[0], vals[1], vals[2]],
            (vals[3], [vals[4], vals[5], vals[6]]),
        )
        //(three_sim::points_mean(points), (0.0, [1.0, 0.0, 0.0]))
    }

    pub fn calc_cost(
        position: [f64; 3],
        rotation: (f64, [f64; 3]),
        position_guess: [f64; 3],
        rotation_guess: (f64, [f64; 3]),
    ) -> f64 {
        position
            .iter()
            .zip(position_guess.iter())
            .map(|(p, g)| (p - g) * (p - g))
            .sum::<f64>()
            + (rotation.0 - rotation_guess.0) * (rotation.0 - rotation_guess.0)
            + rotation
                .1
                .iter()
                .zip(rotation_guess.1.iter())
                .map(|(r, g)| (r - g) * (r - g))
                .sum::<f64>()
    }
}

pub fn generate_random() -> ([f64; 3], (f64, [f64; 3])) {
    let mut rng = rand::thread_rng();
    let axis = [
        rng.gen_range(-1.0, 1.0),
        rng.gen_range(-1.0, 1.0),
        rng.gen_range(-1.0, 1.0),
    ];
    let len = ((axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]) as f64).sqrt();
    (
        [
            rng.gen_range(-2.0, 2.0),
            rng.gen_range(-2.0, 2.0),
            rng.gen_range(-1.0, 1.0),
        ],
        quaternion::axis_angle(
            [axis[0] / len, axis[1] / len, axis[2] / len],
            PI * (1.0 + rng.gen_range(-0.3, 0.3)),
        ),
    )
}

pub fn read_nnet(lines: Vec<String>) -> Option<NeuralNetwork> {
    if lines.iter().next()? != " NNET" {
        return None;
    }
    let geometry = lines
        .iter()
        .skip(1)
        .next()?
        .split(",")
        .map(|s| s.parse::<u32>().unwrap())
        .collect::<Vec<u32>>();
    if geometry.len() < 2 || lines.iter().skip(2).next()?.len() != 0 {
        return None;
    }
    let mut lines = lines.iter().skip(3);
    let mut c = Vec::<Vec<Vec<f64>>>::new();
    let mut b = Vec::<Vec<f64>>::new();
    for ni in 1..geometry.len() {
        let mut cl = Vec::<Vec<f64>>::new();
        let mut bl = Vec::<f64>::new();
        for _ in 0..geometry[ni] {
            let cb = lines
                .next()?
                .split(",")
                .map(|s| s.parse::<f64>().unwrap())
                .collect::<Vec<f64>>();
            if cb.len() != (geometry[ni - 1] + 1) as usize {
                return None;
            }
            cl.push(
                cb.iter()
                    .take(geometry[ni - 1] as usize)
                    .cloned()
                    .collect::<Vec<f64>>(),
            );
            bl.push(
                cb.iter()
                    .skip(geometry[ni - 1] as usize)
                    .cloned()
                    .next()
                    .unwrap(),
            );
        }
        c.push(cl);
        b.push(bl);
        lines.next();
    }
    Some(NeuralNetwork { c, b })
}

pub fn save_nnet(nnet: &NeuralNetwork) -> Vec<String> {
    let mut geometry = vec![nnet.c[0][0].len().to_string()];
    geometry.extend(nnet.c.iter().map(|cl| cl.len().to_string()));

    let mut lines = Vec::<String>::new();
    lines.push(" NNET".to_string());
    lines.push(geometry.join(","));
    lines.push("".to_string());
    for (cl, bl) in nnet.c.iter().zip(nnet.b.iter()) {
        lines.extend(
            cl.iter()
                .zip(bl.iter())
                .map(|(c, b)| {
                    let mut le = c.iter().map(|cx| cx.to_string()).collect::<Vec<String>>();
                    le.push(b.to_string());
                    le.join(",")
                })
                .collect::<Vec<String>>(),
        );
        lines.push("".to_string());
    }
    lines
}
