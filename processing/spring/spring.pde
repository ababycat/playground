import shiffman.box2d.*;

import toxi.physics2d.*;
import toxi.physics2d.behaviors.*;
import toxi.geom.*;

float len = 10;
float strength = 0.01;

int numParticles = 10;

VerletPhysics2D physics;
Particle p1 = new Particle(new Vec2D(100, 20));
Particle p2 = new Particle(new Vec2D(100, 180));
VerletSpring2D spring = new VerletSpring2D(p1, p2, len, strength);

ArrayList<Particle> particles = new ArrayList<Particle>();

void setup(){
  size(500, 500);
  physics= new VerletPhysics2D();
  physics.addBehavior(new GravityBehavior(new Vec2D(0, 0.5)));
  p1.lock();
  physics.addParticle(p1);
  physics.addParticle(p2);
  
  physics.addSpring(spring);
  
  for(int i=0; i < numParticles; ++i){
    Particle p = new Particle(new Vec2D(i*len, 10));
    physics.addParticle(p);
    particles.add(p);
  }
  for(int i=0; i < numParticles-1; ++i){
    VerletSpring2D spring = new VerletSpring2D(particles.get(i), particles.get(i+1), len, strength);
    physics.addSpring(spring);
  }
  
  Particle head = particles.get(0);
  head.x = 250;
  head.y = 0;
  head.lock();
}

void draw(){
  physics.update();
  
  background(255);
  line(p1.x, p1.y, p2.x, p2.y);
  p1.display();
  p2.display();

  
  stroke(0);
  noFill();
  beginShape();
  for(Particle p : particles){
    vertex(p.x, p.y);
  }
  endShape();
  
  Particle tail = particles.get(numParticles-1);
  tail.display();
  if(mousePressed){
    tail.lock();
    tail.x = mouseX;
    tail.y = mouseY;
    tail.unlock();
  }
}

class Particle extends VerletParticle2D{
  Particle(Vec2D loc){
    super(loc);
  }
  void display(){
    fill(0, 150);
    stroke(0);
    ellipse(x, y, 16, 16);
  }
}
