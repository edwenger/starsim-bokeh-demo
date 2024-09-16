""" 
This is where we'll put panel bokeh visualization helper function logic

The plan is as follows:
   - panels:  infectiousness (left) + immunity (right)
   - scatter: household circle networks on grid-per-household
   - edges: highlight cross-links (during infection) and color edge (by new infection from InfectionLog?)

In terms of interactivity:
   - selected: vaccinate + infect buttons
   - sliders: playback speed, exogeneous background importation rate, household beta, extra-household beta/contacts
   - start with reset-&-run button for fixed-duration simulation
   --> explore unbounded sim.ts open-loop adaptations for highlighting recent buffer of individual states??
"""

import random
import starsim as ss
from household import HouseholdResidence, HouseholdPregnancy, HouseholdNetwork


def init_matriarch_sim():

    people = ss.People(n_agents=1)

    hh_demog = HouseholdResidence()

    sim = ss.Sim(
        people=people, 
        diseases=[],
        networks=[
            ss.PrenatalNet(),
            HouseholdNetwork(),
            ], 
        demographics=[
            HouseholdPregnancy(fertility_rate=500), 
            hh_demog,
            # ss.Deaths(death_rate=30),  # TODO: understand odd behavior
            ],
        n_years=100,
        rand_seed=random.randint(0, 1000))
    
    sim.initialize()

    sim.people.age[0] = 15
    sim.people.female[0] = True
    sim.demographics["householdresidence"].huid[0] = next(hh_demog.uid_gen)

    return sim