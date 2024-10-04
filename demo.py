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

    # if relocate_age is < pregnancy.age_min (15), females will move to a new HH
    # before pregnancy, and thus any newborns will reside in the new HH.
    hh_demog = HouseholdResidence(relocate_age = ss.constant(v=14))

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
        dur=100,  # years
        rand_seed=random.randint(0, 1000),
        # slot_scale=1000 # Won't be needed in the future, but changes on a Starsim branch
    )
    
    sim.init()

    sim.people.age[0] = 15
    sim.people.female[0] = True
    hh_demog = sim.demographics["householdresidence"] # Only use hh_demog create above of copy_inputs=False is passed to ss.Sim.
    hh_demog.huid[0] = next(hh_demog.uid_gen)

    return sim

if __name__ == '__main__':
    sim = init_matriarch_sim()
    sim.run()

    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots()
    n = len(sim.people.age)
    ax.scatter(sim.people.age + 0.5*(np.random.rand(n)-0.5), sim.people.female+0.1*np.random.rand(n))

    major_ticks = np.arange(0, sim.people.age.max(), 5)
    minor_ticks = np.arange(0, sim.people.age.max(), 1)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both', axis='x')

    plt.show()
