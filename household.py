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
import sciris as sc
import starsim as ss

ss_int_ = ss.dtypes.int


class HouseholdPregnancy(ss.Pregnancy):
    
    def make_embryos(self, conceive_uids):
        new_uids = super().make_embryos(conceive_uids)

        hh_demog = self.sim.demographics.get("householdresidence")
        if hh_demog is not None and len(conceive_uids) > 0:
            hh_demog.huid[new_uids] = hh_demog.huid[conceive_uids]
            hh_demog.age_relocate[new_uids] = hh_demog.pars.relocate_age.rvs(new_uids)

        return new_uids
    
    def update_states(self):

        ti = self.sim.ti
        deliveries = self.pregnant & (self.ti_delivery <= ti)

        super().update_states()

        # TODO: if we require PrenatalNet + HouseholdResidence to populate HouseholdNet, 
        # should that bundling (and others in this module) be made explicit (w/ requires=[...]).
        # Note also, this peculiar inheritance is working around not caring about a PostnatalNet presently.
        # More explicitly broadcast/receive event-based pattern might be more robust in general here?
        prenatalnet = self.sim.networks.get("prenatalnet")
        if prenatalnet is not None and len(deliveries) > 0:
            
            # Find the prenatal connections that are ending
            prenatal_ending = prenatalnet.edges.end<=self.sim.ti
            new_infant_uids = prenatalnet.edges.p2[prenatal_ending]

            # Remove pairs from prenatal network
            prenatalnet.end_pairs()
                
            # Add edges betwen newly delivered infant and all other household residents
            householdnet = self.sim.networks.get("householdnetwork")
            if householdnet is not None:
                householdnet.add_new_members(new_infant_uids)


class HouseholdResidence(ss.Demographics):
    
    def __init__(self, pars=None, metadata=None, **kwargs):

        super().__init__()

        self.default_pars(
            relocate_age=ss.uniform(18, 25),
        )
        self.update_pars(pars, **kwargs)

        self.add_states(
            ss.FloatArr("huid", label="Household UID"),
            ss.FloatArr("age_relocate", label="Age of relocation to own household"),
        )

        self.uid_gen = itertools.count(start=0, step=1)  # unique next household index generator

    def update(self):
        sim = self.sim

        relocations = (sim.people.female & (self.age_relocate <= sim.people.age)).uids
        if len(relocations):
            self.age_relocate[relocations] = np.nan
            self.huid[relocations] = [next(self.uid_gen) for _ in range(len(relocations))]

            householdnet = self.sim.networks.get("householdnetwork")
            if householdnet is not None:
                householdnet.remove_uids(relocations)

        emigrations = (sim.people.male & (self.age_relocate <= sim.people.age)).uids
        sim.people.request_death(emigrations)


class HouseholdNetwork(ss.Network):

    def __init__(self, key_dict=None, **kwargs):
        key_dict = sc.mergedicts({'huid': ss_int_}, key_dict)
        super().__init__(key_dict=key_dict, **kwargs)

    def add_new_members(self, new_member_ids):

        hh_demog = self.sim.demographics.get("householdresidence")
        if hh_demog is not None:

            # TODO: is there a vectorized way of doing this?
            for new_member_id in new_member_ids:
                huid = hh_demog.huid[new_member_id]

                # get uid indices of all other household members with same hh_uid
                hh_contacts = self.sim.people.uid[hh_demog.huid == huid]
                hh_contacts = hh_contacts[hh_contacts != new_member_id]

                p1 = np.concatenate((hh_contacts, np.repeat(new_member_id, len(hh_contacts))))
                p2 = p1[::-1]
                self.append(p1=p1, p2=p2, beta=np.ones(len(p1)), huid=np.repeat(huid, len(p1)))
    