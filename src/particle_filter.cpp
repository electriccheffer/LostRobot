/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"
#include <cmath>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include "map.h"
#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  
  num_particles = 20;  // TODO: Set the number of particles
  static std::default_random_engine engine(std::random_device{}()); 

  std::normal_distribution<double> gaussianDistributionX(x,std[0]); 
  std::normal_distribution<double> gaussianDistributionY(y,std[1]);
  std::normal_distribution<double> gaussianDistributionTheta(theta,std[2]);
  for(int particle_index = 0 ;particle_index < num_particles; particle_index++ ){
 	
    Particle newParticle; 
    newParticle.id = particle_index; 
    newParticle.x = gaussianDistributionX(engine); 
    newParticle.y = gaussianDistributionY(engine); 
    newParticle.theta = gaussianDistributionTheta(engine);
    newParticle.weight = 1.0; 
    this->particles.push_back(newParticle);
  
  }
  this->is_initialized = true; 
	  
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  int numberOfParticles = this->particles.size(); 

  static std::default_random_engine engine(std::random_device{}()); 
  
  for(int particleIndex = 0 ; particleIndex < numberOfParticles ; particleIndex++){
  
    Particle &particleReference = this->particles[particleIndex]; 
    std::normal_distribution<double> gaussianDistributionX(particleReference.x,std_pos[0]); 
    std::normal_distribution<double> gaussianDistributionY(particleReference.y,std_pos[1]);
    std::normal_distribution<double> gaussianDistributionTheta(particleReference.theta,
		    						std_pos[2]);
  

    if(yaw_rate != 0){
    
      particleReference.x += (velocity/yaw_rate) * (std::sin(particleReference.theta +
				      				yaw_rate * delta_t)
			      		      -
					      std::sin(particleReference.theta)); 

      particleReference.y += (velocity/yaw_rate) * (std::cos(particleReference.theta)
			      		      -
					      std::cos(particleReference.theta +
				      				yaw_rate*delta_t)); 

     particleReference.theta += yaw_rate * delta_t; 
    
    }
    else{
   	
      particleReference.x += velocity * delta_t * std::cos(particleReference.theta);
      particleReference.y += velocity * delta_t * std::sin(particleReference.theta);
    
    }
    
    particleReference.x += gaussianDistributionX(engine);
    particleReference.y += gaussianDistributionY(engine);
    particleReference.theta += gaussianDistributionTheta(engine);
    
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
   
  int observationSize = observations.size(); 
  for(int observationIndex = 0; observationIndex < observationSize; observationIndex++){
  	
    LandmarkObs &observation = observations[observationIndex];
    int landmarkID = -1; 
    double minimum_distance = INFINITY;  
    int predictionSize = predicted.size(); 

    for(int predictionIndex = 0 ; predictionIndex < predictionSize ; predictionIndex++){

      LandmarkObs &prediction = predicted[predictionIndex]; 
      double distance = std::hypot(observation.x - prediction.x,
		                   observation.y - prediction.y); 
      if(distance < minimum_distance){
      	
      	minimum_distance = distance; 
	landmarkID = prediction.id; 
      }

    }

   observation.id = landmarkID; 
  } 
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
   int numberOfParticles = this->particles.size(); 
   for(int particleIndex = 0 ; particleIndex < numberOfParticles ; particleIndex++){
   
     Particle &particle = this->particles[particleIndex]; 
     
     int sizeOfObservations = observations.size(); 
     std::vector<LandmarkObs> transformedObservations;

     for(int observationIndex = 0 ; observationIndex < sizeOfObservations 
		                  ; observationIndex++){
     
       LandmarkObs observation = observations[observationIndex]; 
       double mapXPosition = particle.x + std::cos(particle.theta) * observation.x -
	              std::sin(particle.theta) * observation.y; 
       double mapYPosition = particle.y + std::sin(particle.theta) * observation.x +
	              std::cos(particle.theta) * observation.y; 
       LandmarkObs transformedObservation; 
       transformedObservation.id = -1; 
       transformedObservation.x = mapXPosition; 
       transformedObservation.y = mapYPosition;
       transformedObservations.push_back(transformedObservation);
     }
  	
     std::vector<Map::single_landmark_s> mapLandmarkList = map_landmarks.landmark_list; 
     std::vector<LandmarkObs> predictedLandmarks;
     int numberOfLandmarks = mapLandmarkList.size(); 
     for(int landmarkIndex = 0 ; landmarkIndex < numberOfLandmarks ; landmarkIndex++){
   	
       Map::single_landmark_s landmark = mapLandmarkList[landmarkIndex]; 
       double distance = std::hypot(particle.x - landmark.x_f, particle.y - landmark.y_f); 
       if(distance <= sensor_range){
      	 LandmarkObs nearestLandmark; 
	 nearestLandmark.id = landmark.id_i;
	 nearestLandmark.x =  landmark.x_f; 
	 nearestLandmark.y = landmark.y_f; 
         predictedLandmarks.push_back(nearestLandmark); 
       
       }
     }  
     this->dataAssociation(predictedLandmarks,transformedObservations);
     

     particle.weight = 1.0; 
     int transformedObservationsSize = transformedObservations.size(); 

     for(int weightLoopIndex = 0 ; weightLoopIndex < transformedObservationsSize 
		     						;weightLoopIndex++ ){
     
       LandmarkObs observation = transformedObservations[weightLoopIndex]; 
       LandmarkObs associatedLandmark; 
       bool found = false; 
       for(LandmarkObs predicted : predictedLandmarks){
         if(predicted.id == observation.id){
	   associatedLandmark = predicted; 
	   found = true; 
	   break; 
	 }
       } 
        
       if (!found){
         continue; 
       }
       double deltaX = observation.x - associatedLandmark.x; 
       double deltaY = observation.y - associatedLandmark.y;
       
       double exponent = ((deltaX*deltaX) / (2*std_landmark[0]*std_landmark[0])) + 
	                   ((deltaY * deltaY)/(2*std_landmark[1]*std_landmark[1])); 
       
       double gaussianNorm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]); 
       
       double observationProbability = gaussianNorm * std::exp(-exponent); 
       particle.weight *= observationProbability;  
     
     }  
   }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::vector<Particle> newParticles; 
  int particleNumber = this->num_particles; 
  std::default_random_engine engine; 
  std::uniform_int_distribution<int> distribution(0,particleNumber-1); 
  int index = distribution(engine); 
  double beta = 0; 

  double maxWeight = *std::max_element(this->weights.begin(),this->weights.end()); 
  std::uniform_real_distribution<double> doubleGenerator(0,2 * maxWeight);
  
  for(int i = 0; i < particleNumber; i++){
    beta += doubleGenerator(engine);
      while(beta > this->weights[index]){
        beta -= this->weights[index];
        index = (index + 1) % particleNumber;
      }
     newParticles.push_back(this->particles[index]);
    }

    this->particles = newParticles;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
