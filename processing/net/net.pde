import shiffman.box2d.*;

import toxi.physics2d.*;
import toxi.physics2d.behaviors.*;
import toxi.geom.*;

float len = 4;
float strength = 0.1;

int net_width = 30;
int net_height = 30;


VerletPhysics2D physics;
ArrayList<Particle> particles = new ArrayList<Particle>();

void setup(){
  size(500, 500);
  physics= new VerletPhysics2D();
  physics.addBehavior(new GravityBehavior(new Vec2D(0, 0.5))); 
  
  for(int i=0; i < net_height; ++i){
    for(int j=0; j < net_width; ++j){
      Particle p = new Particle(new Vec2D(i, j));
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
  
  for(int i=net_height-1, j=0; j <net_width-1; ++j){
      VerletSpring2D spring = new VerletSpring2D(
          particles.get(i*net_width+j), 
          particles.get(i*net_width+j+1), len, strength);
      physics.addSpring(spring);
  }
  
  for(int i=0, j=net_width-1; i <net_height-1; ++i){
      VerletSpring2D spring2 = new VerletSpring2D(
          particles.get(i*net_width+j), 
          particles.get((i+1)*net_width+j), len, strength);
      physics.addSpring(spring2);
  }  

  Particle head = particles.get(0);
  head.x = 100;
  head.y = 100;
  head.lock();
  
  Particle head2 = particles.get(net_width*net_height-net_height);
  head2.x = 400;
  head2.y = 100;
  head2.lock();

  Particle head3 = particles.get(net_height-1);
  head3.x = 100;
  head3.y = 400;
  head3.lock();

  Particle head4 = particles.get(net_width*net_height-1);
  head4.x = 400;
  head4.y = 400;
  head4.lock();
  
  int i = 0;
  int j = 10;
  Particle header = particles.get(i*net_height+j);
  header.x = j*len+100;
  header.y = 100;
  header.lock();
  
  i = 0;
  j = 20;
  header = particles.get(i*net_height+j);
  header.x = j*len+100;
  header.y = 100;
  header.lock();
}

void draw(){
  physics.update();
  
  background(255);
  
  stroke(0);
  noFill();
  //beginShape();
  //for(Particle p : particles){
  //  vertex(p.x, p.y);
  //}
  //endShape();
  
  for(int i=0; i < net_height-1; ++i){
    for(int j=0; j < net_width-1; ++j){
      Particle p1 = particles.get(i*net_width+j);
      Particle p2 = particles.get(i*net_width+j+1);
      Particle p3 = particles.get((i+1)*net_width+j);
      line(p1.x, p1.y, p2.x, p2.y);
      line(p1.x, p1.y, p3.x, p3.y);
    }
  }
  for(int i=net_height-1, j=0; j <net_width-1; ++j){
      Particle p1 = particles.get(i*net_width+j);
      Particle p2 = particles.get(i*net_width+j+1);
      line(p1.x, p1.y, p2.x, p2.y);  
  }
  
  for(int i=0, j=net_width-1; i <net_height-1; ++i){
      Particle p1 = particles.get(i*net_width+j);
      Particle p3 = particles.get((i+1)*net_width+j);
      line(p1.x, p1.y, p3.x, p3.y);  
  }  

  
  Particle tail = particles.get(int(net_height*net_width*0.5));
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
