figure,
hold on

for pendulum_start_degree = [-80, -40, 0, 40, 80]
    sim(simple_pendulum);
    plot(q.data, w.data);
end