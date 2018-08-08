
void setup(){
  size(700, 300);

}

void draw(){

  background(255);
  //strokeCap(PROJECT);
  //strokeWeight(1);
  fill(0);
  cantor(10, 0, 700-20);
}


void cantor(float x, float y, float len){
  if(y<height){
    if(len >=1){ //<>//
      rect(x, y, len, 10);
      
      y+=50;
      cantor(x, y, len/3);
      cantor(x+len*2/3, y, len/3); //<>//
    }
  }
}
