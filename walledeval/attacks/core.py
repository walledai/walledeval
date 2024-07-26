# walledeval/attacks/core.py

from abc import ABC, abstractmethod

from walledeval.attacks.mutators import Mutator
from collections.abc import Iterable

__all__ = [
    "Attack", "UniversalAttack", "PromptAttack",
    "MutatorAttack", "CompositeAttack"
]


class Attack(ABC):
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def attack(self, samples: list[str]) -> list[str]:
        raise NotImplementedError


class UniversalAttack(Attack):
    @abstractmethod
    def single_attack(self, sample: str, others: list[str]) -> list[str]:
        raise NotImplementedError
    
    def attack(self, samples: list[str]) -> list[str]:
        attack_results = []
        for idx in range(len(samples)):
            attack_results.extend(self.single_attack(
                samples[idx],
                samples[:idx]+samples[idx+1:]
            ))
        
        return attack_results


class PromptAttack(Attack):
    @abstractmethod
    def single_attack(self, sample: str) -> list[str]:
        raise NotImplementedError
    
    def attack(self, samples: list[str]) -> list[str]:
        attack_results = []
        for idx in range(len(samples)):
            attack_results.extend(self.single_attack(
                samples[idx]
            ))
        
        return attack_results


class MutatorAttack(PromptAttack):
    def __init__(self, mutator: Mutator):
        super().__init__(mutator.name + "Attack")
        self.mutator = mutator
    
    def single_attack(self, sample: str, **kwargs) -> list[str]:
        return [
            self.mutator(sample, **kwargs)
        ]


class CompositeAttack(Attack):
    def __init__(self, *attacks, stack = True):
        self.attacks: list[Attack] = self.flatten(attacks)
        self.stack = stack
    
    def flatten(self, attacks):
        # Recursively flatten attacks
        new_attacks = []
        
        for attack in attacks:
            if isinstance(attack, CompositeAttack):
                # I don't trust myself so I put this failsafe here
                new_attacks.extend(self.flatten(attack.attacks))
            
            elif (
                isinstance(attack, Iterable) and 
                isinstance(attack[0], Attack)
            ):
                new_attacks.extend(self.flatten(attack))
            
            else:
                new_attacks.append(attack)
        
        return new_attacks
    
    def __len__(self) -> int:
        return len(self.attacks)
    
    def attack(self, samples: list[str]) -> list[str]:
        overall_samples = []
        
        for attack in self.attacks:
            overall_samples.extend(
                attack.attack(samples + (overall_samples if self.stack == True else []))
            )