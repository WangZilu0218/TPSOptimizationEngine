//
// Created by 王子路 on 2023/10/19.
//

#ifndef OPTIMIZE_ENZYMEAUTOGRAD_LOSS_H_
#define OPTIMIZE_ENZYMEAUTOGRAD_LOSS_H_
float calDoseLoss(float *d_dose, float *d_dose_grad, float *d_loss, float dose_value, int size, int sign);
float calDVHLoss(float *d_dose, float *d_dose_grad, float *d_loss, float d1, int size, float v1, int sign);
float calgEUDLoss(float *d_dose, float *d_dose_grad, float target, float a, int size, int sign);
float calUniformDoseLoss(float *d_dose, float *d_dose_grad, float *d_loss, float *d_value, int size);
#endif //OPTIMIZE_ENZYMEAUTOGRAD_LOSS_H_
