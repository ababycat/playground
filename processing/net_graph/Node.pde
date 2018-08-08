class Node extends VerletParticle2D{
  String name;
  boolean visible;
  Node(String n, Vec2D loc, boolean isvisible){
    super(loc);
    name = n;
    visible = isvisible;
  }
  void display(){
    fill(0, 150);
    stroke(0);
    ellipse(x, y, 16, 16);
  }
}
