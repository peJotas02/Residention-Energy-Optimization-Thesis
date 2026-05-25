import gymnasium as gym
from gymnasium import spaces
import numpy as np

class HEMSEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, data, dt_hours=1.0):
        super().__init__()
        self.data = data  # e.g., dict with arrays: price_buy, price_sell, pv, load, etc.
        self.dt = dt_hours
        self.batt_capacity = 15
        self.money_values = []
        self.ev_capacity = 30
        self.ev_use_factor = 10
        self.ev_departure_req = self.ev_capacity
        self.alpha = 1/50
        self.beta = 1 / 50



        # Action space (28 total action)
        self.bat_levels = np.array([-5.0, -4, -3, -2, -1, 0.0, 1, 2, 3, 4, 5.0], dtype=np.float32)
        self.ev_levels = np.array([0.0, 0.8, 1.6, 2.4, 3, 3.6], dtype=np.float32)

        #Example
        #                | EV Indexes
        #Battery indexes |     0      |     1      |     2      |     3
        # 0              |   act 0    |   act 1    |   act 2    |   act 3
        # 1              |   act 4    |   act 5    |   act 6    |   act 7
        # 2              |   act 8    |   act 9    |   act 10   |   act 11
        # 3              |   act 12   |   act 13   |   act 14   |   act 15

        self.action_space = spaces.Discrete(len(self.bat_levels) * len(self.ev_levels))

        # Observation space: [is_ev_connected, soc_bat, soc_ev, hour, price_buy, price_sell, pv[t-1], load[t-1], price_buy0, price_buy1, ...price_sell 0, price_sell1, ..., price sell23] (kWh) (sek/ KWh)
        max_buy_price = max(self.data['price_buy'])
        min_buy_price = min(self.data['price_buy'])

        low_obs_array = [ 0, 0.0, 0.0, 0.0, min_buy_price, 0 ,0 ,-1 ,-1]
        self.high_obs_array = [1, 15.0, 30, 23.0, max_buy_price, max(self.data['actual_pv']), max(self.data['actual_load']), 1, 1]

        # Variables for the prices at each hour of the day ---ADD REALISTIC BOUNDS LATER, INSTEAD OF -INF, INF---
        for i in range(24):
            self.high_obs_array.append(max_buy_price)
            low_obs_array.append(min_buy_price)

        self.observation_space = spaces.Box(
            low=np.array(low_obs_array, dtype=np.float32),
            high=np.array(self.high_obs_array, dtype=np.float32),
            dtype=np.float32
        )

        self.reset(t=0, soc_bat=7.5, soc_ev=15)

    def reset(self, t, soc_bat, soc_ev, seed=None, options=None):
        super().reset(seed=seed)
        if t >= len(self.data["price_buy"]):
            self.t = 0
            self.soc_ev = self.ev_capacity / 2
            self.soc_bat = self.batt_capacity / 2  # random.randint(0, 15)
        else:
            self.soc_ev = soc_ev
            self.soc_bat = soc_bat
            self.t = t

        self.ep_len = 0

        # Condition to check ev_connection
        self.is_ev_connected = int(self.data["is_ev_connected"][self.t])

        pv = self.data["actual_pv"][self.t]
        load = self.data["actual_load"][self.t]

        # Simple energy balance
        net_energy_demand = load - pv
        grid_import = max(0.0, net_energy_demand)
        grid_export = max(0.0, - net_energy_demand)
        price_buy = self.data["price_buy"][self.t]
        price_sell = self.data["price_sell"][self.t]

        self.money_spent = -(grid_import * price_buy) + (grid_export * price_sell)

        obs = self._get_obs()
        info = {'time_step': self.t, 'grid import': grid_import, 'grid export': grid_export, 'money_spent': self.money_spent}
        return obs, info

    def decode_action(self, a: int):
        n_ev = len(self.ev_levels)
        i_bat = a // n_ev
        i_ev = a % n_ev
        return float(self.bat_levels[i_bat]), float(self.ev_levels[i_ev])

    def step(self, action, ep_len):

        self.t += 1
        self.ep_len += 1
        P_bat, P_ev = self.decode_action(action)

        # Clip to constraints due to proximity with SOC limits
        #Battery clipping
        P_bat_high = (self.batt_capacity - self.soc_bat) / self.dt  # max charge allowed this step
        P_bat_low = -self.soc_bat / self.dt  # max discharge allowed this step
        P_bat = float(np.clip(P_bat, P_bat_low, P_bat_high))

        # EV clipping
        P_ev_high = (self.ev_capacity - self.soc_ev) / self.dt  # max charge allowed this step
        P_ev_low = -self.soc_ev / self.dt  # max discharge allowed this step
        P_ev = float(np.clip(P_ev, P_ev_low, P_ev_high))

        price_buy = self.data["price_buy"][self.t]
        price_sell = self.data["price_sell"][self.t]
        pv = self.data["actual_pv"][self.t]
        load = self.data["actual_load"][self.t]

        # Simple energy balance
        net_energy_demand = load - pv + (P_bat + P_ev) * self.dt
        grid_import = max(0.0, net_energy_demand)
        grid_export = max(0.0, - net_energy_demand)

        self.money_spent = -(grid_import * price_buy) + (grid_export * price_sell)

        # Update SOCs (very simplified)
        is_ev_connected = self.data["is_ev_connected"][self.t]
        self.soc_bat = self.soc_bat + P_bat * self.dt
        self.soc_ev = self.soc_ev + P_ev * self.dt * int(is_ev_connected)

        # Clipping just to be sure nothing weird happens
        self.soc_bat = float(np.clip(self.soc_bat, 0.0, self.batt_capacity))
        self.soc_ev = float(np.clip(self.soc_ev, 0.0, self.ev_capacity))

        prev_connected = bool(self.data["is_ev_connected"][max(0, self.t - 1)])
        now_connected = bool(self.data["is_ev_connected"][self.t])

        if prev_connected and not now_connected:
            self.soc_ev = self.soc_ev - self.ev_use_factor

        # Penalty factors (Sum money_spent per day average = -92, ev_penalty average per day = 79)
        hour = self.t % 24
        k = 1
        max_price = max(self.data["price_buy"][self.t - hour: self.t - hour + 24])

        #alpha = (max_price ** 4) / ( (self.avg_load - self.avg_pv) * self.beta )
        alpha = abs(self.beta * self.money_spent)

        # Ev continuous penalty for not having the EV at the desired charge
        ev_penalty = - (self.ev_departure_req - self.soc_ev) * alpha

        # Urgency factor to increase penalty near departure and negate penalty when EV is not conncected
        if 6 <= hour <= 10 and is_ev_connected:
            urgency_penalty = k * (hour - 6)**2 + 1

        elif not is_ev_connected:
            urgency_penalty = 0

        else:
            urgency_penalty = 1

        ev_penalty *= urgency_penalty

        # Final Reward
        reward = self.money_spent + ev_penalty

        terminated = (self.t >= len(self.data["actual_pv"]) - 1) or (self.ep_len >= ep_len - 1)
        truncated = False

        obs = self._get_obs()
        info = {'grid import': grid_import, 'grid export': grid_export,
                'money_spent': self.money_spent, 'time step': self.t, 'soc_bat': self.soc_bat, 'soc_ev': self.soc_ev,
                'ev_penalty': ev_penalty,}

        return obs, reward, terminated, truncated, info

    def get_norm_factors(self):
        #just return the upper limits deifined for the Observation space since these accuratly represent the max values of each variable
        return self.high_obs_array


    def _get_obs(self):
        hour = self.t % 24                          #Current hour of the day
        first_hour = self.t - hour                  #First hour of the day 00:00

        n = len(self.data["price_buy"])

        #prices for the full 24h hours of the day
        price_buy_24h = self.data['price_buy'][first_hour: first_hour + 24]

        #Base obs array without the 24h prices
        obs_array = np.array([
            self.is_ev_connected,
            self.soc_bat,
            self.soc_ev,
            hour,
            self.data["price_buy"][self.t],
            self.data["actual_pv"][self.t],
            self.data["actual_load"][self.t],
            self.data['cos_day'][self.t],
            self.data['sin_day'][self.t],
        ], dtype=np.float32)

        obs = np.concatenate((obs_array, price_buy_24h)).astype(np.float32)
        assert obs.shape[0] == self.observation_space.shape[0], (obs.shape, self.observation_space.shape)

        return obs

