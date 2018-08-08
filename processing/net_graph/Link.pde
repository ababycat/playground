class Link extends VerletSpring2D{
  boolean visible;
  Node n1;
  Node n2;
  Link(Node p1, Node p2, float len, float strength, boolean isvisible){
    super(p1, p2, len, strength);
    n1 = p1;
    n2 = p2;
    visible = isvisible;
  }
  void display(){
    if(visible){
      stroke(0);
      line(n1.x(), n1.y(), n2.x(), n2.y());
    }
  }
}
