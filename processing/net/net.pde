import shiffman.box2d.*;

import toxi.physics2d.*;
import toxi.physics2d.behaviors.*;
import toxi.geom.*;

float len = 10;
float strength = 0.01;

int net_width = 10;
int net_height = 10;


VerletPhysics2D physics;
ArrayList<Particle> particles = new ArrayList<Particle>();

void setup(){
  size(500, 500);
  physics= new VerletPhysics2D();
  physics.addBehavior(new GravityBehavior(new Vec2D(0, 0.5))); 
  
  for(int i=0; i < net_height; ++i){
    for(int j=0; j < net_width; ++j){
      Particle p = new Particle(new Vec2D(j, i));
      particles.add(p);
      physics.addParticle(p);
    }
  }
  
  for(int i=0; i < net_height-1; ++i){
    for(int j=0; j < net_width-1; ++j){      
      VerletSpring2D spring = new VerletSpring2D(
          particles.get(i*net_width+j), 
          particles.get(i*net_width+j+1), len, strength);
      physics.addSpring(spring);
      VerletSpring2D spring2 = new VerletSpring2D(
          particles.get(i*net_width+j), 
          particles.get((i+1)*net_width+j), len, strength);
      physics.addSpring(spring2);
    }
  }

  Particle head = particles.get(0);
  head.x = 250;
  head.y = 0;
  head.lock();
  
  Particle head2 = particles.get(net_width*net_height-net_height);
  head2.x = 250;
  head2.y = 0;
  head2.lock();
  
  
}

void draw(){
  physics.update();
  
  background(255);
  
  stroke(0);
  noFill();
  beginShape();
  for(Particle p : particles){
    vertex(p.x, p.y);
  }
  endShape();
  
  Particle tail = particles.get(net_height*net_width-1);
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
