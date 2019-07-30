float MIN_LIVE_LEN = 120.0;
float MAX_LENGTH = 200.0;
float MIN_LENGTH = 150.0;
float MAX_RADIUS_RATE = 0.8;
float MIN_RADIUS_RATE = 0.5;

float LEN_DECAY = 0.99;
float CENTER_DIST_DECAY = 0.95;

color START_COLOR = #6495ED;
color END_COLOR = #FF8C00;


class MovingArrow
{
  float arrow_len;
  float center_dist;
  float angle;
  float cx, cy;
  color arrow_color;
  float radius = sqrt(pow(width/2, 2) + pow(height/2, 2));
  
  MovingArrow(float arrow_len, float center_dist, float angle, float cx, float cy){
    this.arrow_len = arrow_len;
    this.center_dist = center_dist;
    this.angle = angle;
    this.cx = cx;
    this.cy = cy;
    this.arrow_color = lerpColor(START_COLOR, END_COLOR, center_dist/radius);
  }
  
  void update(){
    this.arrow_len *= LEN_DECAY;
    this.center_dist *= CENTER_DIST_DECAY;
    this.arrow_color = lerpColor(START_COLOR, END_COLOR, center_dist/radius);
  }
  
  float len(){
    return this.arrow_len;
  }
  
  void display(){
    pushMatrix();
    translate(cx, cy);
    rotate(angle);
    stroke(this.arrow_color);
    line(0, this.center_dist, 0, this.center_dist + this.arrow_len);
    popMatrix();
  }
  
  boolean alive(){
    if(len() < MIN_LIVE_LEN){
      return false;
    }else{
      return true;
    }
  }
};


class ArrowSystem
{
  ArrayList<MovingArrow> system = new ArrayList();
  
  float cx, cy;
  
  int max_num;
  int random_add_num;
  float random_p;
  float radius = sqrt(pow(width/2, 2) + pow(height/2, 2));
  
  ArrowSystem(int max_num, 
              int random_add_num, 
              float random_p, 
              float cx,
              float cy){
    this.max_num = max_num;
    this.random_add_num = random_add_num;
    this.random_p = random_p;
    this.cx = cx;
    this.cy = cy;
  }
  
  void update(){
    IntList rm_list = new IntList();
    for(int i=0; i < system.size(); ++i){
      MovingArrow arrow = system.get(i); 
      arrow.update();
       if(!arrow.alive()){
          rm_list.append(i);
       }
    }
    
    // remove object which not alive
    for(int i=rm_list.size()-1; i >=0; --i){
    
      system.remove(rm_list.get(i));
    }
    
    
    // add object
    for(int i=0; i < this.random_add_num; ++i){
      if(system.size() >= this.max_num){
        break;
      }
      
      if(random(0, 1) < this.random_p){
        MovingArrow arrow = new MovingArrow(
                                     random(MIN_LENGTH, MAX_LENGTH), 
                                     random(MAX_RADIUS_RATE*radius, MIN_RADIUS_RATE*radius), 
                                     random(0, TWO_PI),
                                     this.cx, this.cy);
        system.add(arrow);
      }
    }
  }
  
  void display(){
    for(MovingArrow arrow : system){
       arrow.display();
    }  
  }
};
