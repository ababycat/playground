class ParticalSystem{
  VerletPhysics2D physics= new VerletPhysics2D();
  ArrayList<Node> nodes = new ArrayList<Node>();
  ArrayList<Link> links = new ArrayList<Link>();
  ParticalSystem(){
    //physics.addBehavior(new GravityBehavior(new Vec2D(0, 0.5)));
    physics.setWorldBounds(new Rect(0, 0, width, height));
  }
  
  Node get_node_by_name(String name){
    int idx = 0;
    for(; idx < nodes.size(); ++idx){
      if(nodes.get(idx).name == name){
        break;      
      }      
    }
    return nodes.get(idx);
  }
  
  void add_node(String name, Vec2D loc, boolean isvisible){
    Node n = new Node(name, loc, isvisible);
    physics.addParticle(n);
    nodes.add(n);
  }
  
  void remove_node_by_name(String name){
    int idx = 0;
    for(; idx < nodes.size(); ++idx){
      if(nodes.get(idx).name == name){
        break;      
      }      
    }
    physics.removeParticle(nodes.get(idx));
    nodes.remove(idx);
  }
  
  void add_link(Node n1, Node n2, float len, float strength, boolean visible){
    Link l = new Link(n1, n2, len, strength, visible);
    physics.addSpring(l);    
    links.add(l);
  }
  
  void remove_link_by_name(String name1, String name2){
    for(Link l : links){
      if(((l.n1.name == name1)&&(l.n2.name == name2))||(((l.n1.name == name2)&&(l.n2.name == name1)))){
        physics.removeSpring(l);
        links.remove(l);
        break;
      }
    }
  }
  
  void update(){
    physics.update();
  }
  
  void display(){
    
    for(Node n : nodes){
      n.display();
    } 
    for(Link l : links){
      l.display();
    }
  }
}
