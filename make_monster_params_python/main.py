from enum import Enum
from typing import Tuple
import json
import random
import time
import uuid


class Status:
    def __init__(self, max_hp: int, atk: int, def_: int, luc: int, speed: int):
        """
        Constructor

        Parameters
        ----------
        max_hp : int
            Max HP
        atk : int
            Attack
        def_ : int
            Defense
        luc : int
            Luck(0~100)
        speed : int
            Speed
        """
        self.max_hp = max_hp
        self.atk = atk
        self.def_ = def_
        self.luc = luc
        self.speed = speed


class Config:
    def __init__(self,
                 gene_num: int,
                 selection_rate: float,
                 mutation_rate: float,
                 generation_max: int,
                 turn_min: int,
                 turn_max: int,
                 min_status: dict,
                 max_status: dict,
                 character_status: dict,
                 show_progress: bool):
        """
        Constructor

        Parameters
        ----------
        gene_num : int
            Number of genes
        selection_rate : float
            Selection rate(0~1.0)
        mutation_rate : float
            Mutation rate(0~1.0)
        generation_max : int
            Maximum number of generations
        turn_min : int
            Minimum number of turns
        turn_max : int
            Maximum number of turns
        min_status : dict
            Minimum status value
        max_status : dict
            Maximum status value
        character_status : dict
            Player's status
        show_progress : bool
            True if you want to display the progress
        """
        self.gene_num = gene_num
        self.selection_rate = selection_rate
        self.mutation_rate = mutation_rate
        self.generation_max = generation_max
        self.turn_min = turn_min
        self.turn_max = turn_max
        self.min_status = self.__convert_dict_to_status(min_status)
        self.max_status = self.__convert_dict_to_status(max_status)
        self.character_status = self.__convert_dict_to_status(character_status)
        self.show_progress = show_progress

    @classmethod
    def read_from_json(cls):
        """ Loads Config from config.json. """
        with open('config.json', 'r') as f:
            data = json.load(f)
        return cls(**data)

    @staticmethod
    def __convert_dict_to_status(dict_: dict) -> Status:
        converted_status = {}
        for key, value in dict_.items():
            if key.startswith("def") and value is not None:
                converted_status[key.replace("def", "def_")] = value
            else:
                converted_status[key] = value
        return Status(**converted_status)


Fitness = float

WORST_SCORE = 1_000_000.0


class Unit:
    def __init__(self, is_monster: bool, status: Status):
        self.id = uuid.uuid4()
        self.is_monster = is_monster
        self.max_hp = status.max_hp
        self.hp = status.max_hp
        self.atk = status.atk
        self.def_ = status.def_
        self.luc = status.luc
        self.speed = status.speed

    @classmethod
    def character_from(cls, status: Status):
        return cls(False, status)

    @classmethod
    def monster_from(cls, status: Status):
        return cls(True, status)

    def to_s(self) -> str:
        return f"{{ hp={self.hp}/{self.max_hp} atk={self.atk} def={self.def_} luc={self.luc} speed={self.speed} }}"


class BattleResult(Enum):
    WIN = 1
    GAME_OVER = 2
    DRAW = 3


def battle(character: Unit, monster: Unit, config: Config) -> Tuple[BattleResult, int]:
    """
    Returns
    -------
    Tuple[BattleResult, int]
        A tuple of BattleResult and the number of turns completed.
    """
    units = sorted([character, monster], key=lambda u: u.speed, reverse=True)

    hp_map = {u.id: u.hp for u in units}

    turn = 0
    while True:
        turn += 1
        if turn > config.turn_max:
            result = BattleResult.DRAW
            break

        stop = None
        for attacker in units:
            if hp_map[attacker.id] <= 0:
                # Already dead
                continue

            # Target of an attack
            defender = character if attacker.is_monster else monster

            hp = hp_map[defender.id]
            hp -= calc_damage(attacker, defender)
            hp = max(0, hp)
            hp_map[defender.id] = hp

            if hp_map[defender.id] <= 0:
                if defender.is_monster:
                    stop = BattleResult.WIN
                    break
                else:
                    stop = BattleResult.GAME_OVER
                    break

        if stop is not None:
            result = stop
            break

    character.hp = hp_map[character.id]

    return result, turn


def calc_damage(attacker: Unit, defender: Unit) -> int:
    luc = attacker.luc / 100.0
    c = 2.0 if random.random() <= luc else 1.0
    atk = attacker.atk
    def_ = defender.def_
    damage = int((atk - def_ * 0.5) * c)
    return max(0, damage)


class FitnessFunc:
    def __init__(self, config: Config):
        self.config = config

    def calc(self, monster_status: Status) -> Tuple[Fitness, BattleResult]:
        character = Unit.character_from(self.config.character_status)
        monster = Unit.monster_from(monster_status)

        result, turn = battle(character, monster, self.config)

        # Percentage of HP remaining
        hp_ratio = float(character.hp) / float(character.max_hp)

        # Fitness
        if result == BattleResult.GAME_OVER:
            score = WORST_SCORE
        elif result == BattleResult.DRAW:
            score = WORST_SCORE - 1.0
        elif turn < self.config.turn_min:
            score = WORST_SCORE - 1.0
        else:
            score = hp_ratio

        if self.config.show_progress:
            print("-------------------------------")
            print(f"score={score} {result} turn={turn}")
            print(f"character={character.to_s()}")
            print(f"monster={monster.to_s()}")
            print("-------------------------------")

        return score, result


# Parameters of Gene
GeneParams = Status


class Gene:
    def __init__(self, params: GeneParams):
        self.id = uuid.uuid4()
        self.fitness = WORST_SCORE
        self.params = params

    def to_s(self) -> str:
        return (f"fitness={self.fitness} "
                f"{{ max_hp={self.params.max_hp} atk={self.params.atk} def={self.params.def_} luc={self.params.luc} speed={self.params.speed} }}")


class GeneMask:
    def __init__(self):
        """ Generates True or False mask for each field at random. """
        self.max_hp = random.choice([True, False])
        self.atk = random.choice([True, False])
        self.def_ = random.choice([True, False])
        self.luc = random.choice([True, False])
        self.speed = random.choice([True, False])


class Mutation:
    def __init__(self, min: Status, max: Status):
        self.min = min
        self.max = max

    def new_at_random(self) -> Gene:
        """ Generates individuals at random. """
        params = GeneParams(
            max_hp=random.randint(self.min.max_hp, self.max.max_hp),
            atk=random.randint(self.min.atk, self.max.atk),
            def_=random.randint(self.min.def_, self.max.def_),
            luc=random.randint(self.min.luc, self.max.luc),
            speed=random.randint(self.min.speed, self.max.speed)
        )
        return Gene(params)

    def mutated(self, gene: Gene, rate: float) -> Gene:
        """ Mutates. """
        return Gene(GeneParams(
            max_hp=gene.params.max_hp if random.random() > rate else random.randint(self.min.max_hp, self.max.max_hp),
            atk=gene.params.atk if random.random() > rate else random.randint(self.min.atk, self.max.atk),
            def_=gene.params.def_ if random.random() > rate else random.randint(self.min.def_, self.max.def_),
            luc=gene.params.luc if random.random() > rate else random.randint(self.min.luc, self.max.luc),
            speed=gene.params.speed if random.random() > rate else random.randint(self.min.speed, self.max.speed)
        ))


class GeneticAlgorithm:
    def __init__(self, config: Config):
        self.config = config
        self.genes = []

    def selection_num(self) -> int:
        """ Returns the number of individuals to leave to the next generation. """
        return int(self.config.gene_num * self.config.selection_rate)

    def best_score(self) -> Fitness:
        """ Returns the best score. """
        return self.genes[0].fitness

    def head(self, num) -> list:
        """ Returns the specified number of individuals from the head. """
        return self.genes[:num]

    def sample(self) -> Gene:
        """ Returns one randomly selected individual. """
        return random.choice(self.genes)

    def exec(self):
        """ Executes the genetic algorithm. """
        gene_num = self.config.gene_num

        selection_num = self.selection_num()

        mutation = Mutation(min=self.config.min_status, max=self.config.max_status)

        # Generates initial generation
        for _ in range(gene_num):
            self.genes.append(mutation.new_at_random())

        ff = FitnessFunc(self.config)

        # Calculates the fitness each individual
        for gene in self.genes:
            gene.fitness = ff.calc(gene.params)[0]

        # Current best score
        cur_best_score = WORST_SCORE
        # Current generation
        generation = 1
        while True:
            # Sorts in descending order of goodness of fitness
            self.genes = sorted(self.genes, key=lambda gene: gene.fitness)

            best_score = self.best_score()
            if cur_best_score != best_score:
                cur_best_score = best_score
                print(f"{generation}G: {best_score}")

            if generation >= self.config.generation_max:
                # Finish
                break

            # Selection
            # Leaves superior individuals to the next generation
            new_genes = self.head(selection_num)

            # Generates individuals by crossover and mutation for the number of unselected
            while len(new_genes) < gene_num:
                # Select parents
                parents = self.select_parents()
                # Crossover
                children = self.crossover(parents[0], parents[1])

                # Child1
                # Mutation
                g1 = mutation.mutated(children[0], self.config.mutation_rate)
                # Calculates the fitness
                g1.fitness = ff.calc(g1.params)[0]
                new_genes.append(g1)
                if len(new_genes) == gene_num:
                    break

                # Child2
                # Mutation
                g2 = mutation.mutated(children[1], self.config.mutation_rate)
                # Calculates the fitness
                g2.fitness = ff.calc(g2.params)[0]
                new_genes.append(g2)
                if len(new_genes) == gene_num:
                    break

            self.genes = new_genes

            generation += 1

    def select_parents(self) -> Tuple[Gene, Gene]:
        """ Selects parents at random. """
        p1 = self.sample()
        p2 = p1
        while p1.id == p2.id:
            p2 = self.sample()
        return p1, p2

    def crossover(self, p1: Gene, p2: Gene) -> Tuple[Gene, Gene]:
        """
        Parent 1 and Parent 2 are crossed to produce new individuals.

        Parameters
        ----------
        p1 : Gene
            Parent 1
        p2 : Gene
            Parent 2
        """
        mask = GeneMask()

        params1 = GeneParams(
            max_hp=p2.params.max_hp if mask.max_hp else p1.params.max_hp,
            atk=p2.params.atk if mask.atk else p1.params.atk,
            def_=p2.params.def_ if mask.def_ else p1.params.def_,
            luc=p2.params.luc if mask.luc else p1.params.luc,
            speed=p2.params.speed if mask.speed else p1.params.speed
        )
        params2 = GeneParams(
            max_hp=p1.params.max_hp if mask.max_hp else p2.params.max_hp,
            atk=p1.params.atk if mask.atk else p2.params.atk,
            def_=p1.params.def_ if mask.def_ else p2.params.def_,
            luc=p1.params.luc if mask.luc else p2.params.luc,
            speed=p1.params.speed if mask.speed else p2.params.speed
        )
        return Gene(params1), Gene(params2)


def main():
    config = Config.read_from_json()

    ga = GeneticAlgorithm(config)

    start = time.time()

    ga.exec()

    end = time.time() - start

    # Test play with highest score parameter
    best_gene = ga.head(1)[0]
    ff = FitnessFunc(config)
    win_count = 0
    simulation_count = 100
    for _ in range(simulation_count):
        _, result = ff.calc(best_gene.params)
        if result == BattleResult.WIN:
            win_count += 1

    print(f"elapsed={end:.3f}sec")
    print(f"{[g.to_s() for g in ga.head(5)]}")
    print(f"win={win_count / simulation_count * 100.0}%")


if __name__ == "__main__":
    main()
