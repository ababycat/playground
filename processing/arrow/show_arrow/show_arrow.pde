
ArrowSystem arrow_sys;
int noise_x = 100;
int noise_y = 10000;

int step = 100;

void setup(){
  //fullScreen();
  size(500, 500);
  arrow_sys = new ArrowSystem(300, 300, 0.1, width/2, height/2);
  
}

void draw(){
  stroke(133);
  background(255);
  arrow_sys.update();
  arrow_sys.display();
  
  
  //arrow_sys.cx = noise(noise_x)*width;
  //arrow_sys.cy = noise(noise_y)*height;
 
  //if(--step == 0){
  //  step = 100;
  //  noise_x++;
  //  noise_y++;
  //  if(noise_x > 200){
  //     noise_x = 100;
  //  }
  //  if(noise_y > 10200){
  //    noise_y = 10000;
  //  }
  //}
  
  //println(arrow_sys.cx, arrow_sys.cy);
}


void mouseMoved() {
  arrow_sys.cx = mouseX;
  arrow_sys.cy = mouseY;
  save("arrow.jpg");
}
