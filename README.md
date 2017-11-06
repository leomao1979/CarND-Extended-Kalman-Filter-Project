# Extended Kalman Filter Project Starter Code
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Self-Driving Car Engineer Nanodegree Program

In this project you will utilize a kalman filter to estimate the state of a moving object of interest with noisy lidar and radar measurements. Passing the project requires obtaining RMSE values that are lower that the tolerance outlined in the project rubric.

[//]: # (Image References)
[result_dataset1]: images/result_dataset1.png
[result_dataset2]: images/result_dataset2.png

## [Rubric](https://review.udacity.com/#!/rubrics/748/view) Points

### Results

RMSE with dataset1

| Item     | Value       |  
|:--------:|:-----------:|
| X        | 0.0973      |
| Y        | 0.0855      |
| VX       | 0.4513      |
| VY       | 0.4399      |

![Result of dataset1][result_dataset1]

RMSE with dataset2

| Item     | Value       |  
|:--------:|:-----------:|
| X        | 0.0726      |
| Y        | 0.0967      |
| VX       | 0.4579      |
| VY       | 0.4966      |

![Result of dataset2][result_dataset2]

### First measurement
Initialize state vector according to sensor type when receive the first measurement.
```
if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Convert radar from polar to cartesian coordinates and initialize state.
    ekf_.x_ = tools.ConvertToCartesianCoordinates(measurement_pack.raw_measurements_);
} else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
    ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
}

```

Though radar gives velocity data in the form of the range rate rho_dot, a radar measurement does not contain enough information to determine the state variable velocities vx and vy. Use the radar measurements rho and theta to initialize the state variable locations px and py.

```
VectorXd Tools::ConvertToCartesianCoordinates(const VectorXd& x_polar) {
    VectorXd x(4);
    float rho     = x_polar(0);
    float theta   = x_polar(1);
    // float rho_dot = x_polar(2);
    float px = rho * cos(theta);
    float py = rho * sin(theta);
    x << px, py, 0, 0;
    return x;
}
```

### Predict
1) Update the state transition matrix F according to the new elapsed time.
```
// dt - expressed in seconds
float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;   
float dt_2 = dt * dt;
float dt_3 = dt_2 * dt;
float dt_4 = dt_3 * dt;

// Modify the F matrix so that the time is integrated
ekf_.F_(0, 2) = dt;
ekf_.F_(1, 3) = dt;
```
2) Set the process covariance matrix Q (noise_ax = 9 and noise_ay = 9)
```
ekf_.Q_ = MatrixXd(4, 4);
float noise_ax = 9, noise_ay = 9;
ekf_.Q_ <<  dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
            0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
            dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
            0, dt_3/2*noise_ay, 0, dt_2*noise_ay;
```

3) Predict
```
x_ = F_ * x_;
MatrixXd Ft = F_.transpose();
P_ = F_ * P_ * Ft + Q_;
```

### Update
1) Update measurement matrix H and measurement covariance matrix R according
For Laser type:
```
ekf_.H_ << 1, 0, 0, 0,
            0, 1, 0, 0;

ekf_.R_ << 0.0225, 0,
            0, 0.0225;
```

For Radar type, the function that maps x vector to polar coordinates is non-linear, so use Jacobian matrix:
```
ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
ekf_.R_ << 0.09, 0, 0,
            0, 0.0009, 0,
            0, 0, 0.09;
```

2) Update state vector and state covariance matrix

For Laser Type:
```
VectorXd z_predict = H_ * x_;
VectorXd y = z - z_predict;
MatrixXd Ht = H_.transpose();
MatrixXd S = H_ * P_ * Ht + R_;
MatrixXd Si = S.inverse();
MatrixXd K = P_ * Ht * Si;

//new estimate
x_ = x_ + (K * y);
long x_size = x_.size();
MatrixXd I = MatrixXd::Identity(x_size, x_size);
P_ = (I - K * H_) * P_;
```

For Radar type:
```
Tools tools;
VectorXd z_predict = tools.ConvertToPolarCoordinates(x_);
VectorXd y = z - z_predict;
// normalize angel
while (y(1) > PI) {
    y(1) -= 2 * PI;
}
while(y(1) < -PI) {
    y(1) += 2 * PI;
}

MatrixXd Ht = H_.transpose();
MatrixXd S = H_ * P_ * Ht + R_;
MatrixXd Si = S.inverse();
MatrixXd K = P_ * Ht * Si;

// New estimate
x_ = x_ + (K * y);
long x_size = x_.size();
MatrixXd I = MatrixXd::Identity(x_size, x_size);
P_ = (I - K * H_) * P_;
```
