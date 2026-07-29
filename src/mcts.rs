use crate::{
    game::{Board, Move, Outcome},
    model::PolicyValueModel,
};

#[derive(Clone, Copy)]
pub struct SearchConfig {
    pub simulations: usize,
    pub cpuct: f32,
}
impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            simulations: 400,
            cpuct: 1.5,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Candidate {
    pub mv: Move,
    pub visits: u32,
    pub q: f32,
    pub prior: f32,
}
struct Node {
    board: Board,
    children: Vec<Edge>,
    expanded: bool,
}
struct Edge {
    mv: Move,
    prior: f32,
    visits: u32,
    value_sum: f32,
    child: Option<usize>,
}

pub fn search(board: &Board, model: &PolicyValueModel, cfg: SearchConfig) -> Vec<Candidate> {
    let mut nodes = vec![Node {
        board: board.clone(),
        children: vec![],
        expanded: false,
    }];
    expand(&mut nodes, 0, model);
    for _ in 0..cfg.simulations {
        simulate(&mut nodes, 0, model, cfg.cpuct);
    }
    let mut out: Vec<_> = nodes[0]
        .children
        .iter()
        .map(|e| Candidate {
            mv: e.mv,
            visits: e.visits,
            q: if e.visits == 0 {
                0.0
            } else {
                e.value_sum / e.visits as f32
            },
            prior: e.prior,
        })
        .collect();
    out.sort_by_key(|x| std::cmp::Reverse(x.visits));
    out
}
fn expand(nodes: &mut Vec<Node>, idx: usize, model: &PolicyValueModel) -> f32 {
    if let Some(out) = nodes[idx].board.outcome() {
        return match out {
            Outcome::Draw => 0.0,
            Outcome::Win(p) => {
                if p == nodes[idx].board.to_move() {
                    1.0
                } else {
                    -1.0
                }
            }
        };
    }
    let (policy, value) = model.evaluate(&nodes[idx].board);
    nodes[idx].children = policy
        .into_iter()
        .map(|(mv, prior)| Edge {
            mv,
            prior,
            visits: 0,
            value_sum: 0.0,
            child: None,
        })
        .collect();
    nodes[idx].expanded = true;
    value
}
fn simulate(nodes: &mut Vec<Node>, idx: usize, model: &PolicyValueModel, cpuct: f32) -> f32 {
    if let Some(out) = nodes[idx].board.outcome() {
        return match out {
            Outcome::Draw => 0.0,
            Outcome::Win(p) => {
                if p == nodes[idx].board.to_move() {
                    1.0
                } else {
                    -1.0
                }
            }
        };
    }
    if !nodes[idx].expanded {
        return expand(nodes, idx, model);
    }
    let total = nodes[idx]
        .children
        .iter()
        .map(|e| e.visits)
        .sum::<u32>()
        .max(1) as f32;
    let mut best = 0;
    let mut score = f32::NEG_INFINITY;
    for (i, e) in nodes[idx].children.iter().enumerate() {
        let q = if e.visits == 0 {
            0.0
        } else {
            e.value_sum / e.visits as f32
        };
        let s = q + cpuct * e.prior * total.sqrt() / (1.0 + e.visits as f32);
        if s > score {
            score = s;
            best = i
        }
    }
    let child = if let Some(c) = nodes[idx].children[best].child {
        c
    } else {
        let mut b = nodes[idx].board.clone();
        b.play(nodes[idx].children[best].mv);
        let c = nodes.len();
        nodes.push(Node {
            board: b,
            children: vec![],
            expanded: false,
        });
        nodes[idx].children[best].child = Some(c);
        c
    };
    let value = -simulate(nodes, child, model, cpuct);
    let e = &mut nodes[idx].children[best];
    e.visits += 1;
    e.value_sum += value;
    value
}
