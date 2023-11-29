use std::collections::HashMap;
use std::{cmp, fs};
use rand::Rng;
use rand::distributions::{Distribution, Uniform};
use std::time::{Instant};
use rand::rngs::ThreadRng;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
struct Status {
    /// Max HP
    max_hp: i32,
    /// Attack
    atk: i32,
    /// Defense
    def: i32,
    /// Luck(0~100)
    luc: i32,
    /// Speed
    speed: i32,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
struct Config {
    /// Number of genes
    gene_num: i32,
    /// Selection rate(0~1.0)
    selection_rate: f32,
    /// Mutation rate(0~1.0)
    mutation_rate: f32,
    /// Maximum number of generations
    generation_max: i32,
    /// Minimum number of turns
    turn_min: i32,
    /// Maximum number of turns
    turn_max: i32,
    /// Minimum status value
    min_status: Status,
    /// Maximum status value
    max_status: Status,
    /// Player's status
    character_status: Status,
    /// True if you want to display the progress
    show_progress: bool,
}

impl Config {
    /// Loads Config from config.json.
    fn read_from_json() -> Self {
        let content = fs::read_to_string("config.json")
            .expect("Failed to load config.json");
        serde_json::from_str(&content).unwrap()
    }
}

type Fitness = f32;

const WORST_SCORE: Fitness = 1_000_000.0;

#[derive(Clone, Copy, Debug)]
struct Unit {
    id: Uuid,
    is_monster: bool,
    max_hp: i32,
    hp: i32,
    atk: i32,
    def: i32,
    luc: i32,
    speed: i32,
}

impl Unit {
    fn character_from(status: &Status) -> Unit {
        Unit {
            id: Uuid::new_v4(),
            is_monster: false,
            max_hp: status.max_hp,
            hp: status.max_hp,
            atk: status.atk,
            def: status.def,
            luc: status.luc,
            speed: status.speed,
        }
    }

    fn monster_from(status: &Status) -> Unit {
        Unit {
            id: Uuid::new_v4(),
            is_monster: true,
            max_hp: status.max_hp,
            hp: status.max_hp,
            atk: status.atk,
            def: status.def,
            luc: status.luc,
            speed: status.speed,
        }
    }

    fn to_s(&self) -> String {
        format!("{{ hp={}/{} atk={} def={} luc={} speed={} }}",
                self.hp,
                self.max_hp,
                self.atk,
                self.def,
                self.luc,
                self.speed)
    }
}

#[derive(PartialEq, Debug)]
enum BattleResult {
    Win,
    GameOver,
    Draw,
}

/// # Returns
/// Returns a tuple of BattleResult and the number of turns completed.
fn battle(character: &mut Unit, monster: Unit, config: &Config) -> (BattleResult, i32) {
    let mut rng = rand::thread_rng();

    let mut units: Vec<Unit> = vec![
        character.clone(),
        monster.clone(),
    ];
    units.sort_by(|a, b| b.speed.cmp(&a.speed));    // speed desc

    let mut hp_map: HashMap<Uuid, i32> = HashMap::new();
    for u in units.iter() {
        hp_map.insert(u.id, u.hp);
    }

    let mut turn = 0;
    let result = loop {
        turn += 1;
        if turn > config.turn_max {
            break BattleResult::Draw;
        }

        let mut stop: Option<BattleResult> = None;

        for attacker in units.iter() {
            if *hp_map.get(&attacker.id).unwrap() <= 0 {
                // Already dead
                continue;
            }

            // Target of an attack
            let defender = if attacker.is_monster {
                character.clone()
            } else {
                monster.clone()
            };

            let mut hp = *hp_map.get(&defender.id).unwrap();
            hp -= calc_damage(&attacker, &defender, &mut rng);
            hp = cmp::max(0, hp);
            hp_map.insert(defender.id, hp);

            if *hp_map.get(&defender.id).unwrap() <= 0 {
                if defender.is_monster {
                    stop = Some(BattleResult::Win);
                    break;
                } else {
                    stop = Some(BattleResult::GameOver);
                    break;
                }
            }
        }

        match stop {
            Some(res) => break res,
            None => {}
        };
    };

    character.hp = *hp_map.get(&character.id).unwrap();

    (result, turn)
}

fn calc_damage(attacker: &Unit, defender: &Unit, rng: &mut ThreadRng) -> i32 {
    let luc = attacker.luc as f32 / 100.0;
    let c = if rng.gen::<f32>() <= luc { 2.0 } else { 1.0 };
    let atk = attacker.atk as f32;
    let def = defender.def as f32;
    let damage = ((atk - def * 0.5) * c).floor() as i32;
    cmp::max(0, damage)
}

struct FitnessFunc {
    config: Config,
}

impl FitnessFunc {
    fn new(config: Config) -> FitnessFunc {
        FitnessFunc {
            config,
        }
    }

    pub fn calc(&self, monster_status: &Status) -> (Fitness, BattleResult) {
        let mut character = Unit::character_from(&self.config.character_status);
        let monster = Unit::monster_from(&monster_status);

        let (result, turn) = battle(&mut character, monster, &self.config);

        // Percentage of HP remaining
        let hp_ratio = character.hp as f32 / character.max_hp as f32;

        // Fitness
        let score = if result == BattleResult::GameOver {
            WORST_SCORE
        } else if result == BattleResult::Draw {
            WORST_SCORE - 1.0
        } else if turn < self.config.turn_min {
            WORST_SCORE - 1.0
        } else {
            hp_ratio
        };

        if self.config.show_progress {
            println!("-------------------------------");
            println!("score={} {:?} turn={}", score, result, turn);
            println!("character={}", character.to_s());
            println!("monster={}", monster.to_s());
            println!("-------------------------------");
        }

        (score, result)
    }
}

/// Parameters of Gene
type GeneParams = Status;

#[derive(Clone, Copy, Debug)]
struct Gene {
    id: Uuid,
    fitness: Fitness,
    params: GeneParams,
}

impl Gene {
    fn new(params: GeneParams) -> Self {
        Gene {
            id: Uuid::new_v4(),
            fitness: WORST_SCORE,
            params,
        }
    }

    fn to_s(&self) -> String {
        format!("fitness={} {{ max_hp={} atk={} def={} luc={} speed={} }}",
                self.fitness,
                self.params.max_hp,
                self.params.atk,
                self.params.def,
                self.params.luc,
                self.params.speed)
    }
}

struct GeneMask {
    max_hp: bool,
    atk: bool,
    def: bool,
    luc: bool,
    speed: bool,
}

impl GeneMask {
    /// Generates true or false mask for each field at random.
    fn new() -> Self {
        let mut rng = rand::thread_rng();
        GeneMask {
            max_hp: rng.gen(),
            atk: rng.gen(),
            def: rng.gen(),
            luc: rng.gen(),
            speed: rng.gen(),
        }
    }
}

struct Mutation {
    min: Status,
    max: Status,
}

impl Mutation {
    /// Generates individuals at random.
    fn new_at_random(&self) -> Gene {
        let mut rng = rand::thread_rng();
        let hp = Uniform::from(self.min.max_hp..=self.max.max_hp);
        let atk = Uniform::from(self.min.atk..=self.max.atk);
        let def = Uniform::from(self.min.def..=self.max.def);
        let luc = Uniform::from(self.min.luc..=self.max.luc);
        let speed = Uniform::from(self.min.speed..=self.max.speed);

        let params = GeneParams {
            max_hp: hp.sample(&mut rng),
            atk: atk.sample(&mut rng),
            def: def.sample(&mut rng),
            luc: luc.sample(&mut rng),
            speed: speed.sample(&mut rng),
        };
        Gene::new(params)
    }

    /// Mutates.
    fn mutated(&self, gene: &Gene, rate: f32) -> Gene {
        let mut rng = rand::thread_rng();
        let hp = Uniform::from(self.min.max_hp..=self.max.max_hp);
        let atk = Uniform::from(self.min.atk..=self.max.atk);
        let def = Uniform::from(self.min.def..=self.max.def);
        let luc = Uniform::from(self.min.luc..=self.max.luc);
        let speed = Uniform::from(self.min.speed..=self.max.speed);
        let mut params = gene.params.clone();

        // rng.gen::<f32>() -> 0 <= p < 1.0
        if rng.gen::<f32>() <= rate {
            params.max_hp = hp.sample(&mut rng);
        }
        if rng.gen::<f32>() <= rate {
            params.atk = atk.sample(&mut rng);
        }
        if rng.gen::<f32>() <= rate {
            params.def = def.sample(&mut rng);
        }
        if rng.gen::<f32>() <= rate {
            params.luc = luc.sample(&mut rng);
        }
        if rng.gen::<f32>() <= rate {
            params.speed = speed.sample(&mut rng);
        }

        Gene::new(params)
    }
}

struct GeneticAlgorithm {
    config: Config,
    genes: Vec<Gene>,
}

impl GeneticAlgorithm {
    fn new(config: Config) -> Self {
        let genes: Vec<Gene> = vec![];
        GeneticAlgorithm {
            config,
            genes,
        }
    }

    /// Returns the number of individuals to leave to the next generation.
    fn selection_num(&self) -> usize {
        (self.config.gene_num as f32 * self.config.selection_rate).floor() as usize
    }

    /// Returns the best score.
    fn best_score(&self) -> Fitness {
        self.genes[0].fitness
    }

    /// Returns the specified number of individuals from the head.
    fn head(&self, num: usize) -> Vec<Gene> {
        (&self.genes[..num]).to_vec()
    }

    /// Returns one randomly selected individual.
    fn sample(&self) -> Gene {
        let mut rng = rand::thread_rng();
        let idx = Uniform::from(0..self.genes.len());
        self.genes[idx.sample(&mut rng)].clone()
    }

    /// Executes the genetic algorithm.
    fn exec(&mut self) {
        let gene_num = self.config.gene_num as usize;

        let selection_num = self.selection_num();

        let mutation = Mutation {
            min: self.config.min_status.clone(),
            max: self.config.max_status.clone(),
        };

        // Generates initial generation
        for _ in 0..gene_num {
            self.genes.push(mutation.new_at_random());
        }

        let ff = FitnessFunc::new(self.config.clone());

        // Calculates the fitness each individual
        for gene in self.genes.iter_mut() {
            gene.fitness = ff.calc(&gene.params).0;
        }

        // Current best score
        let mut cur_best_score = WORST_SCORE;
        // Current generation
        let mut generation = 1;
        loop {
            // Sorts in descending order of goodness of fitness
            self.genes.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());

            let best_score = self.best_score();
            if cur_best_score != best_score {
                cur_best_score = best_score;
                println!("{}G: {}", generation, best_score);
            }

            if generation >= self.config.generation_max {
                // Finish
                break;
            }

            // Selection
            // Leaves superior individuals to the next generation
            let mut new_genes = self.head(selection_num);

            // Generates individuals by crossover and mutation for the number of unselected
            loop {
                // Selects parents
                let parents = self.select_parents();
                // Crossover
                let children = self.crossover(&parents.0, &parents.1);

                // Child1
                // Mutation
                let mut g1 = mutation.mutated(&children.0, self.config.mutation_rate);
                // Calculates the fitness
                g1.fitness = ff.calc(&g1.params).0;
                new_genes.push(g1);
                if new_genes.len() == gene_num {
                    break;
                }

                // Child2
                // Mutation
                let mut g2 = mutation.mutated(&children.1, self.config.mutation_rate);
                // Calculates the fitness
                g2.fitness = ff.calc(&g2.params).0;
                new_genes.push(g2);
                if new_genes.len() == gene_num {
                    break;
                }
            }
            self.genes = new_genes;
            assert_eq!(self.genes.len(), gene_num);

            generation += 1;
        }
    }

    /// Selects parents at random.
    fn select_parents(&self) -> (Gene, Gene) {
        let p1 = self.sample();
        let p2 = loop {
            let p2 = self.sample();
            if p1.id != p2.id {
                break p2;
            }
        };
        (p1, p2)
    }

    /// Parent 1 and Parent 2 are crossed to produce new individuals.
    ///
    /// # Arguments
    /// * `p1` - Parent 1
    /// * `p2` - Parent 2
    fn crossover(&self, p1: &Gene, p2: &Gene) -> (Gene, Gene) {
        let mask = GeneMask::new();

        let params1 = GeneParams {
            max_hp: if mask.max_hp { p2.params.max_hp } else { p1.params.max_hp },
            atk: if mask.atk { p2.params.atk } else { p1.params.atk },
            def: if mask.def { p2.params.def } else { p1.params.def },
            luc: if mask.luc { p2.params.luc } else { p1.params.luc },
            speed: if mask.speed { p2.params.speed } else { p1.params.speed },
        };
        let params2 = GeneParams {
            max_hp: if mask.max_hp { p1.params.max_hp } else { p2.params.max_hp },
            atk: if mask.atk { p1.params.atk } else { p2.params.atk },
            def: if mask.def { p1.params.def } else { p2.params.def },
            luc: if mask.luc { p1.params.luc } else { p2.params.luc },
            speed: if mask.speed { p1.params.speed } else { p2.params.speed },
        };
        (Gene::new(params1), Gene::new(params2))
    }
}

fn main() {
    let config = Config::read_from_json();

    let mut ga = GeneticAlgorithm::new(config.clone());

    let start = Instant::now();

    ga.exec();

    let end = start.elapsed();

    // Test play with the highest score parameter
    let best_gene = ga.head(1)[0];
    let ff = FitnessFunc::new(config);
    let mut win_count = 0;
    let simulation_count = 100;
    for _ in 0..simulation_count {
        let (_, result) = ff.calc(&best_gene.params);
        if result == BattleResult::Win {
            win_count += 1;
        }
    }

    println!("elapsed={}.{:03}sec", end.as_secs(), end.subsec_nanos() / 1_000_000);
    println!("{:#?}", ga.head(5).iter().map(|g| g.to_s()).collect::<Vec<String>>());
    println!("win={}%", win_count as f32 / simulation_count as f32 * 100.0);
}
