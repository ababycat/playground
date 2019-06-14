

void setup(){
  size(300, 300);
}

float noiseScale = 0.02;

void draw() {
  background(0);
  for (int x=0; x < width; x++) {
    float noiseVal = noise((mouseX+x)*noiseScale, mouseY*noiseScale);
    stroke(noiseVal*255);
    line(x, mouseY+noiseVal*80, x, height);
  }
}


//void draw(){
////line(30, 20, 85, 20);
////stroke(126);
////line(85, 20, 85, 75);
////stroke(color(255, 0, 0));
////line(85, 75, 30, 75);
//  //  pushMatrix();
//  //stroke(0);
//  //background(255);
//  //color from = color(255, 0, 0);
//  //color to = color(0, 0, 255);
//  //color interA = lerpColor(from, to, 15.1/15.2);
//  //color interB = lerpColor(from, to, 15.1/15.2);
//  //stroke(interB);
//  //line(85, 75, 30, 75);
//  //popMatrix();
//  //println(interB);
  
    


  
//  //fill(from);
//  //rect(10, 20, 20, 60);
//  //fill(interA);
//  //rect(30, 20, 20, 60);
//  //fill(interB);
//  //rect(50, 20, 20, 60);
//  //fill(to);
//  //rect(70, 20, 20, 60);
//}

//void draw(){
//  background(204);
//  translate(150, 150);
//  for (int i = 0; i < 10; i++) {
//    rotate(PI*0.1);
//    stroke(0);
//    float dist = abs(100);
//    line(0, 0, dist, 0);
//  }
//  println(sqrt(pow(5, 2)+));
//}

//float[] distribution = new float[360];

//void setup() {
//  size(500, 500);
//  for (int i = 0; i < distribution.length; i++) {
//    distribution[i] = int(randomGaussian() * 120);
//  }
//}

//void draw() {
//  background(204);
  
//  translate(width/2, width/2);

//  for (int i = 0; i < distribution.length; i++) {
//    rotate(TWO_PI/distribution.length);
//    stroke(0);
//    float dist = abs(distribution[i]);
//    line(0, 0, dist, 0);
//  }
//}
