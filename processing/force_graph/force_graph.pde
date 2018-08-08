import shiffman.box2d.*;

import toxi.physics2d.*;
import toxi.physics2d.behaviors.*;
import toxi.geom.*;

//import gifAnimation.*;
//GifMaker gifExport;
int num = 5;
float len = 100;
float strength = 0.1;

VerletPhysics2D physics;

Cluster cluster;

void setup(){
  size(500, 500);
  cluster = new Cluster(num, len, new Vec2D(width/2, height/2));
  physics= new VerletPhysics2D();
  //physics.addBehavior(new GravityBehavior(new Vec2D(0, 0.5)));
  physics.setWorldBounds(new Rect(0, 0, width, height));
  for(int i=0; i < cluster.nodes.size(); ++i){
    physics.addParticle(cluster.nodes.get(i));  
  }
  for(int i=0; i < cluster.nodes.size() - 1; ++i){
    VerletParticle2D p1 = cluster.nodes.get(i);
    for(int j=i+1; j < cluster.nodes.size(); ++j){
      VerletParticle2D p2 = cluster.nodes.get(j);
      VerletSpring2D spring = new VerletSpring2D(p1, p2, len, strength);
      physics.addSpring(spring);
    }
  }
  //gifExport = new GifMaker(this, "export.gif");
  //gifExport.setRepeat(0);             // make it an "endless" animation
  //gifExport.setTransparent(0,0,0);    // black is transparent
}

void draw(){
  physics.update();
  
  background(255);
  
  stroke(0);
  noFill();
  for(int i=0; i < cluster.nodes.size() - 1; ++i){
    VerletParticle2D p1 = cluster.nodes.get(i);
    for(int j=i+1; j < cluster.nodes.size(); ++j){
      VerletParticle2D p2 = cluster.nodes.get(j);
      line(p1.x, p1.y, p2.x, p2.y);
    }
  }
  cluster.display();
  
  if(mousePressed){
    Node n = cluster.nodes.get(0);
    n.lock();
    n.x = mouseX;
    n.y = mouseY;
    n.unlock();
  }
    //gifExport.setDelay(1);
    //gifExport.addFrame();
}

void keyPressed(){
  if(key=='s'){
    save("img.png");
    //gifExport.finish();
  }
}

class Node extends VerletParticle2D{
  Node(Vec2D loc){
    super(loc);
  }
  void display(){
    fill(0, 150);
    stroke(0);
    ellipse(x, y, 16, 16);
  }
}

class Cluster{
  ArrayList<Node> nodes = new ArrayList<Node>();
  float diameter;
  Cluster(int n, float d, Vec2D center){
    diameter=d;
    for(int i=0; i < n; ++i){
      println(center.x(), center.y());
      nodes.add(new Node(center.add(Vec2D.randomVector())));
    }
  }
  void display(){
    for(int i=0; i < nodes.size(); ++i){
      nodes.get(i).display();
    }
  }
}
