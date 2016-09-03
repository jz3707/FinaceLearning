def generate_paths(self, fixed_seed=False, day_count=365.):
    if self.time_grid is None:
        self.generate_time_grid()
    M = len(self.time_grid)
    I = self.paths
    paths = np.zeros((M, I))
    paths[0] = self.initial_value
    if not self.correlated:
        rand = sn_random_numbers((1, M, I), fixed_seed=fixed_seed)
        print("rand : ", rand.shape)
    else:
        rand = self.random_numbers
    short_rate = self.discount_curve.short_rate
    for t in range(1, len(self.time_grid)):
        if not self.correlated:
            ran = rand[t]
        else:
            ran = np.dot(self.cholesky_matrix, rand[:, t, :])
            ran = ran[self.rn_set]
        dt = (self.time_grid[t] - self.time_grid[t - 1]).days / day_count
        print("ran shape : ", ran.shape)
        print("rand : ", rand.shape)
        paths[t] = paths[t - 1] * np.exp((short_rate - 0.5 * self.volatility ** 2) * dt +
                                         self.volatility * np.sqrt(dt) * ran)

    self.instrument_values = paths