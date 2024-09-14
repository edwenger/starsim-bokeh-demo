"""
This is matriarch-centric toy demographics.

The plan is as follows:
- Track explicit pregnancy from conception to delivery
- Random draw for next pregnancy if eligible by age + post-partum state
- Children leave the household at a certain age
- Adult females found new household; males leave simulation
- Elderly women die
"""

import itertools
import numpy as np
import starsim as ss


class HouseholdPregnancy(ss.Pregnancy):
    
    def make_embryos(self, conceive_uids):
        new_uids = super().make_embryos(conceive_uids)

        hh_demog = self.sim.demographics.get("householdresidence")

        if hh_demog is not None and len(conceive_uids) > 0:
            hh_demog.huid[new_uids] = hh_demog.huid[conceive_uids]
            hh_demog.ti_relocate[new_uids] = self.sim.ti + hh_demog.pars.relocate_age.rvs(new_uids) - self.sim.people.age[new_uids]

        return new_uids


class HouseholdResidence(ss.Demographics):
    
    uid = itertools.count()  # unique index generator (TODO: adapt to IndexArr pattern??)

    def __init__(self, pars=None, metadata=None, **kwargs):
        super().__init__()
        self.default_pars(
            relocate_age=ss.uniform(18, 25),
        )
        self.update_pars(pars, **kwargs)
        self.add_states(
            ss.FloatArr("huid", label="Household UID"),
            ss.FloatArr("ti_relocate", label="Time of relocation to own household"),
        )

    def update(self):
        sim = self.sim

        relocations = (sim.people.female & (self.ti_relocate <= sim.ti)).uids
        self.ti_relocate[relocations] = np.nan
        self.huid[relocations] = [next(self.uid) for _ in range(len(relocations))]

        emigrations = (~sim.people.female & (self.ti_relocate <= sim.ti)).uids
        sim.people.request_death(emigrations)


    

    