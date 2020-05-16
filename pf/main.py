def particle_filter(motions, measurements, N=500): # I know it's tempting, but don't change N!
	# --------
	#
	# Make particles
	# 

	p = []
	for i in range(N):
		r = robot()
		r.set_noise(bearing_noise, steering_noise, distance_noise)
		p.append(r)

	# --------
	#
	# Update particles
	#     

	for t in range(len(motions)):

		# motion update (prediction)
		p2 = []
		for i in range(N):
			p2.append(p[i].move(motions[t]))
		p = p2

		# measurement update
		w = []
		for i in range(N):
			w.append(p[i].measurement_prob(measurements[t]))

		# resampling
		p3 = []
		index = int(random.random() * N)
		beta = 0.0
		mw = max(w)
		for i in range(N):
			beta += random.random() * 2.0 * mw
			while beta > w[index]:
				beta -= w[index]
				index = (index + 1) % N
			p3.append(p[index])
		p = p3

	return get_position(p)
