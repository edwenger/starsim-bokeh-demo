"""
Port of disease logic from poliosim with reference to original ImmunoInfection model

Famulare et al. 2018 PLOS Biology https://doi.org/10.1371/journal.pbio.2002468
Wong et al. 2023 Virus Evolution https://doi.org/10.1093/ve/vead044

- Infection.pars.beta is mapped to fecal-oral dose parameter
- Infection.rel_sus tracks current_immunity in OPV-equivalent antibody titers
- Infection.rel_trans tracks viral_shed in CID50/g
"""

import numpy as np
import starsim as ss

ss_int_ = ss.dtypes.int


sabin_scale_parameters = dict(WPV=2.3, OPV1=14, OPV2=8, OPV3=18)

strain_take_modifiers= dict(WPV=1.0, OPV1=0.79, OPV2=0.92, OPV3=0.81)  # Based on estimate from Famulare 2018

shed_duration_params = dict(OPV=ss.Pars(u=30.3, delta=1.16, sigma=1.86),
                            WPV=ss.Pars(u=43.0, delta=1.16,sigma=1.69))


class Polio(ss.Infection):
        
    def __init__(self, pars=None, *args, **kwargs):
        super().__init__()
        self.default_pars(

            # Required Infection parameters
            init_prev = ss.bernoulli(p=0.01),
            beta = 5e-7,  # fecal-oral dose

            # Immunoinfection default coefficients
            theta_Nabs = ss.Pars(a=4.82, b=-0.30, c=3.31, d=-0.32),
            immunity_boost_dist = ss.normal(),  # (loc, scale) set on-the-fly in update_peak_immunity

            shed_duration = shed_duration_params['WPV'],
            shed_duration_dist = ss.lognorm_im(),  # (s, scale) set on-the-fly in update_shed_duration

            immunity_waning = ss.Pars(rate=0.87),

            viral_shedding = ss.Pars(eta=1.65, v=0.17, epsilon=0.32),
            peak_cid50 = ss.Pars(k=0.056, Smax=6.7, Smin=4.3, tau=12),

            p_transmit = ss.Pars(alpha=0.44, gamma=0.46),

            sabin_scale_parameter = sabin_scale_parameters['WPV'],
            strain_take_modifier = strain_take_modifiers['WPV'],

        )
        self.update_pars(pars=pars, **kwargs)

        # States
        self.add_states(

            ss.FloatArr("prechallenge_immunity", label="Pre-challenge immunity", default=1),
            ss.FloatArr("postchallenge_peak_immunity", label="Post-challenge peak immunity"),
            ss.FloatArr("shed_duration", label="Shed duration"),
            ss.FloatArr("log10_peak_cid50", label="Log10 peak CID50"),

            # TODO: consider tracking "Exposed" state (as in poliosim) and also to avoid divide-by-zero errors in update_viral_shed

            # Inherited from Infection
            # ss.BoolArr('susceptible', default=True, label='Susceptible'),
            # ss.BoolArr('infected', label='Infectious'),
            # ss.FloatArr('ti_infected', label='Time of infection'),

            # Explicitly define appropriate default values + labels for inherited states
            ss.FloatArr('rel_sus', default=1.0, label='Current immunity (titers)'),
            ss.FloatArr('rel_trans', default=0.0, label='Viral shedding (CID50/g)'),
        )

    @property
    def naive(self):
        return self.ti_infected.isnan
    
    @property
    def t_since_last_exposure(self):
        return (self.sim.ti - self.ti_infected) * self.sim.dt * 365  # TODO: update when sim.unit = 'day' is implemented

    @property
    def current_immunity(self):
        return self.rel_sus

    @property
    def viral_shed(self):
        return self.rel_trans

    def update_pre(self):
        super().update_pre()

        # waning of post-exposure immunity in non-naive individuals
        self.update_current_immunity(**self.pars.immunity_waning)

        # update state variables for cleared infections    
        recovered = (self.infected & (self.t_since_last_exposure >= self.shed_duration)).uids
        self.infected[recovered] = False
        self.susceptible[recovered] = True
        self.viral_shed[recovered] = 0

        # calculate shedding in infectious individuals
        self.update_viral_shed(**self.pars.viral_shedding)
    
    def p_transmit(self, rel_trans_src, rel_sus_target, beta_per_dt):
    
        # Dose corresponds to: viral_shed[src] * fecal_oral_dose
        dose = rel_trans_src * beta_per_dt * 365  # TODO: update when sim.unit = 'day' is implemented

        pars = self.pars
        alpha = pars.p_transmit['alpha']
        gamma = pars.p_transmit['gamma']

        prob_infection = (1 - (1 + dose / pars.sabin_scale_parameter) ** (-alpha * (rel_sus_target) ** -gamma)) * pars.strain_take_modifier

        return prob_infection

    # TODO: remove if/when this abstraction of p_transmit out of make_new_cases is incorporated in base Infection class
    def make_new_cases(self):
        """
        Add new cases of module, through transmission, incidence, etc.
        
        Common-random-number-safe transmission code works by mapping edges onto
        slots.
        """
        new_cases = []
        sources = []
        networks = []
        betamap = self._check_betas()

        for i, (nkey,net) in enumerate(self.sim.networks.items()):
            if not len(net):
                break

            nbetas = betamap[nkey]
            edges = net.edges

            rel_trans = self.rel_trans.asnew(self.infectious * self.rel_trans)
            rel_sus   = self.rel_sus.asnew(self.susceptible * self.rel_sus)
            p1p2b0 = [edges.p1, edges.p2, nbetas[0]]
            p2p1b1 = [edges.p2, edges.p1, nbetas[1]]
            for src, trg, beta in [p1p2b0, p2p1b1]:

                # Skip networks with no transmission
                if beta == 0:
                    continue

                # Calculate probability of a->b transmission.
                beta_per_dt = net.beta_per_dt(disease_beta=beta, dt=self.sim.dt)
                # p_transmit = rel_trans[src] * rel_sus[trg] * beta_per_dt
                p_transmit = self.p_transmit(rel_trans[src], rel_sus[trg], beta_per_dt)

                # Generate a new random number based on the two other random numbers
                rvs_s = self.rng_source.rvs(src)
                rvs_t = self.rng_target.rvs(trg)
                rvs = ss.combine_rands(rvs_s, rvs_t)
                
                new_cases_bool = rvs < p_transmit
                new_cases.append(trg[new_cases_bool])
                sources.append(src[new_cases_bool])
                networks.append(np.full(np.count_nonzero(new_cases_bool), dtype=ss_int_, fill_value=i))
                
        # Tidy up
        if len(new_cases) and len(sources):
            new_cases = ss.uids.cat(new_cases)
            new_cases, inds = new_cases.unique(return_index=True)
            sources = ss.uids.cat(sources)[inds]
            networks = np.concatenate(networks)[inds]
        else:
            new_cases = np.empty(0, dtype=int)
            sources = np.empty(0, dtype=int)
            networks = np.empty(0, dtype=int)

        if len(new_cases):
            self._set_cases(new_cases, sources)
            
        return new_cases, sources, networks

    def set_prognoses(self, uids, source_uids=None):
        super().set_prognoses(uids, source_uids)

        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = self.sim.ti

        self.prechallenge_immunity[uids] = self.current_immunity[uids]
        self.update_peak_immunity(uids, **self.pars.theta_Nabs)
        self.current_immunity[uids] = np.fmax(1, self.postchallenge_peak_immunity[uids])

        self.update_shed_duration(uids, **self.pars.shed_duration)

    def update_peak_immunity(self, uids, a=4.82, b=-0.30, c=3.31, d=-0.32):
        '''immunity immediately post infection'''
        Nabs = self.prechallenge_immunity[uids]
        means = a + b * np.log2(Nabs)
        stdevs = np.sqrt(np.maximum(c + d * np.log2(Nabs), 0))
        self.pars.immunity_boost_dist.set(loc=means, scale=stdevs)
        theta_Nabs = np.exp(self.pars.immunity_boost_dist.rvs(n=uids))
        # prevent immunity from decreasing due to challenge
        self.postchallenge_peak_immunity[uids] = self.prechallenge_immunity[uids] * np.maximum(1, theta_Nabs)

    def update_shed_duration(self, uids, u=30.3, delta=1.16, sigma=1.86):
        """probability of shedding given Nab at time t (days post infection);
        assumes that individual was infected at t = 0; time is measured in days
        Equation S1 in Famulare 2018 PLOS Bio paper
        delta_t = time (days) since last infection -- survival curve follows lognormal distribution"""

        mu = np.log(u) - np.log(delta) * np.log2(self.prechallenge_immunity[uids])
        std = np.log(sigma)

        # NB: ss.Dist classes implicitly handle numpy-to-scipy.stats parameter-convention
        #     conversion via sync_pars() during __init__() or set() with (mean=mu, sigma=std) such that:
        # the shape parameter (s) is the same as the std
        # the scale parameter is the same as the e^(mu)
        self.pars.shed_duration_dist.set(sigma=std, mean=mu)
        self.shed_duration[uids] = self.pars.shed_duration_dist.rvs(n=uids)

    def update_current_immunity(self, rate=0.87):
        '''immunity after t months have passed since exposure'''

        # Waning after 30 days since last exposure (post-challenge pre-waning value set in set_prognoses)
        waning = (~self.naive & (self.t_since_last_exposure >= 30)).uids

        self.current_immunity[waning] = np.fmax(1, self.postchallenge_peak_immunity[waning] * ( (self.t_since_last_exposure[waning]/30) ** -rate ))

    def update_viral_shed(self, eta=1.65, v=0.17, epsilon=0.32):
        '''virus shed per gram, time of infection exposure in days'''
        
        infected_uids = self.infected.uids

        self.update_log10_peak_cid50(infected_uids, **self.pars.peak_cid50)

        t_inf = self.t_since_last_exposure[infected_uids]
        log_t_inf = np.log(t_inf)
        exponent = eta - (0.5*v**2) - ((log_t_inf - eta)**2) / (2 * (v + epsilon*log_t_inf)**2)
        predicted_concentration = 10**self.log10_peak_cid50[infected_uids] * np.exp(exponent) / t_inf

        self.viral_shed[infected_uids] = np.fmax(10**2.6, predicted_concentration)   # Set floor on viral shed to be at least 398

    def update_log10_peak_cid50(self, infected_uids, k=0.056, Smax=6.7, Smin=4.3, tau=12):
        '''
        returns the peak log10(cid50/g) given prior immunity
        model state's age in years, Wes' equation's age in months
        '''

        age_in_months = self.sim.people.age[infected_uids] * 12

        peak_cid50_naiive = np.where(
            age_in_months >= 6, 
            (Smax - Smin) * np.exp((7 - age_in_months) / tau) + Smin,  # if 6 months or older
            Smax)  # else

        self.log10_peak_cid50[infected_uids] = (1 - k * np.log2(self.prechallenge_immunity[infected_uids])) * peak_cid50_naiive


class OPV1(Polio):
    def __init__(self, pars=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_pars(
            sabin_scale_parameter=sabin_scale_parameters['OPV1'],
            strain_take_modifier=strain_take_modifiers['OPV1'],
            shed_duration=shed_duration_params['OPV'],
        )
        self.update_pars(pars=pars, **kwargs)


class OPV2(Polio):
    def __init__(self, pars=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_pars(
            sabin_scale_parameter=sabin_scale_parameters['OPV2'],
            strain_take_modifier=strain_take_modifiers['OPV2'],
            shed_duration=shed_duration_params['OPV'],
        )
        self.update_pars(pars=pars, **kwargs)


class OPV3(Polio):
    def __init__(self, pars=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_pars(
            sabin_scale_parameter=sabin_scale_parameters['OPV3'],
            strain_take_modifier=strain_take_modifiers['OPV3'],
            shed_duration=shed_duration_params['OPV'],
        )
        self.update_pars(pars=pars, **kwargs)


if __name__ == "__main__":

    import starsim as ss

    people = ss.People(n_agents=1)
    
    sim = ss.Sim(
        people=people, 
        diseases=[Polio(init_prev=1.0)],
        networks=[], 
        demographics=[],
        n_years=1,
        use_aging=True,
        dt=1/365.,
        rand_seed=123,
    )
    sim.initialize()

    sim.people.age[0] = 15
    sim.people.female[0] = True

    sim.step()
