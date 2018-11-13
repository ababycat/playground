import shiffman.box2d.*;

import toxi.physics2d.*;
import toxi.physics2d.behaviors.*;
import toxi.geom.*;

//import gifAnimation.*;
//GifMaker gifExport;
int num = 5;
float len = 100;
float strength = 0.01;

ParticalSystem ps;

void setup(){
  size(500, 500);
  ps = new ParticalSystem();
  
  ps.add_node("base_begin", Vec2D(0.0, height/2.), true);
  
  ps.add_node("n1", Vec2D.randomVector(), true);

}

void draw(){
  
  background(255);
  ps.update();
  ps.display();
  
  //physics.update();
  
  //background(255);
  
  //stroke(0);
  //noFill();
  //for(int i=0; i < cluster.nodes.size() - 1; ++i){
  //  VerletParticle2D p1 = cluster.nodes.get(i);
  //  for(int j=i+1; j < cluster.nodes.size(); ++j){
  //    VerletParticle2D p2 = cluster.nodes.get(j);
  //    line(p1.x, p1.y, p2.x, p2.y);
  //  }
  //}
  //cluster.display();
  
  //if(mousePressed){
  //  Node n = cluster.nodes.get(0);
  //  n.lock();
  //  n.x = mouseX;
  //  n.y = mouseY;
  //  n.unlock();
  //}

}

void keyPressed(){
  if(key=='s'){
    save("img.png");
  }
}
