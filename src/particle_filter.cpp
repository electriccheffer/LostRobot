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
  
  num_particles = 2000;  // TODO: Set the number of particles
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
    this->weights.push_back(newParticle.weight);  
  }
  this->is_initialized = true; 
  std::cout << "INIT FINISHED" << std::endl;  
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

  std::cout << "---STARTING PREDICTION---" << std::endl;
  static std::default_random_engine engine(std::random_device{}());

  std::normal_distribution<double> noiseX(0.0, std_pos[0]);
  std::normal_distribution<double> noiseY(0.0, std_pos[1]);
  std::normal_distribution<double> noiseTheta(0.0, std_pos[2]);

  int idx = 0;
  for (auto &p : particles) {
    double prev_x = p.x;
    double prev_y = p.y;
    double prev_theta = p.theta;

    if (std::fabs(yaw_rate) > 1e-4) {
      double new_theta = p.theta + yaw_rate * delta_t;
      double v_over_yaw = velocity / yaw_rate;
      p.x += v_over_yaw * (sin(new_theta) - sin(p.theta));
      p.y += v_over_yaw * (cos(p.theta) - cos(new_theta)); 
      p.theta = new_theta;
    } else {
      p.x += velocity * delta_t * cos(p.theta);
      p.y += velocity * delta_t * sin(p.theta);
    }

        p.x += noiseX(engine);
        p.y += noiseY(engine);
        p.theta += noiseTheta(engine);

        idx++;
    }
    
std::cout << "---END PREDICTION---" << std::endl;



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
 std::cout << "---RUNNING UPDATE WEIGHTS----" << std::endl;
double std_x = std_landmark[0];
    double std_y = std_landmark[1];

    for (int i = 0; i < num_particles; ++i) {
        Particle &p = particles[i];
        double weight = 1.0;

        for (const auto &obs : observations) {
            double x_map = p.x + cos(p.theta) * obs.x - sin(p.theta) * obs.y;
            double y_map = p.y + sin(p.theta) * obs.x + cos(p.theta) * obs.y;

            LandmarkObs nearest;
            double min_dist_sq = sensor_range * sensor_range; 
            for (const auto &lm : map_landmarks.landmark_list) {
                double dx = x_map - lm.x_f;
                double dy = y_map - lm.y_f;
                double dist_sq = dx*dx + dy*dy;
                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    nearest = {lm.id_i, lm.x_f, lm.y_f};
                }
            }

            double dx = x_map - nearest.x;
            double dy = y_map - nearest.y;
            double gauss_norm = 1.0 / (2.0 * M_PI * std_x * std_y);
            double prob = gauss_norm * exp(-(dx*dx/(2*std_x*std_x) + dy*dy/(2*std_y*std_y)));

            weight *= (prob > 0) ? prob : 1e-10; 
        }

        p.weight = weight;
        weights[i] = weight;
    }

    double sum_w = std::accumulate(weights.begin(), weights.end(), 0.0);
    for (auto &p : particles) p.weight /= sum_w;
    for (auto &w : weights) w /= sum_w;
    std::cout << "---FINISHING UPDATE WEIGHTS----" << std::endl;  
   }


void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http:/en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

    std::cout << "-----RUNNING RESAMPLE---" << std::endl;

    if (this->weights.empty()) {
        std::cerr << "WEIGHTS ARE EMPTY!" << std::endl;
        return;
    }

    std::vector<Particle> newParticles;
    int particleNumber = this->num_particles;
    newParticles.reserve(particleNumber);

    double weightSum = std::accumulate(this->weights.begin(), this->weights.end(), 0.0);
    if (weightSum < 1e-300) { 
        for (auto &p : this->particles) {
            p.weight = 1.0 / particleNumber;
        }
        this->weights.assign(particleNumber, 1.0 / particleNumber);
        weightSum = 1.0;
    } else {
        for (int i = 0; i < particleNumber; ++i) {
            this->weights[i] /= weightSum;
        }
    }

    std::default_random_engine engine(std::random_device{}());
    std::uniform_int_distribution<int> indexDist(0, particleNumber - 1);
    int index = indexDist(engine);
    double maxWeight = *std::max_element(this->weights.begin(), this->weights.end());
    std::uniform_real_distribution<double> betaDist(0.0, 2.0 * maxWeight);
    double beta = 0.0;

    for (int i = 0; i < particleNumber; i++) {
        beta += betaDist(engine);
        while (beta > this->weights[index]) {
            beta -= this->weights[index];
            index = (index + 1) % particleNumber;
        }
        Particle p = this->particles[index];

        std::normal_distribution<double> noise(0.0,1e-10);
        p.x += noise(engine);
        p.y += noise(engine);
        p.theta += noise(engine);

        p.weight = 1.0; 
        newParticles.push_back(p);
    }

    this->particles = newParticles;

    std::cout << "Particle positions after resample:" << std::endl;
    for (const auto &p : particles) {
        std::cout << "x=" << p.x
                  << " y=" << p.y
                  << " theta=" << p.theta
                  << std::endl;
    }
    std::cout << "-----FINISHING RESAMPLE---" << std::endl;

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
