#pragma once
#include <stdlib.h>
#include <vector>
#include <math.h>
#include <string.h>
#include <algorithm>
#include <float.h>
#include "gp_mat_utility.h"

using namespace std;

namespace my_gausian_process{

    template <typename T>
    class gausian_process
    {
    private:
	    vector<T> curve_fitting_x;
	    vector<T> curve_fitting_t;
        vector<T> covariance;
	    vector<T> predict_point;
        T sigma;
	    T noise;
	    unsigned sampling_cnt;
        const unsigned predict_cnt;

    public:
	    gausian_process(const unsigned predict_cnt_v);
	    gausian_process(vector<T>& a,vector<T>& b,const unsigned predict_cnt_v);
	    gausian_process(const gausian_process<T>& a);
	    ~gausian_process(){;}
	    T predict(T target);
        void set_noise(T _v){noise = _v;}
        void set_sigma(T _v){sigma = _v;}
        void set_observed_point(vector<T> a){curve_fitting_x = a;}
        void set_observed_value(vector<T> a){curve_fitting_t = a;}
        T get_predict_point_value(const unsigned index){return predict_point[index];}
        void calc_covariance();
        void predict_meanmatrix(vector<T>& result);
        void predict_covfmatrix(vector<T>& result);
        void create_predict_point();
    };

    class bayes_optimize
    {
    private:
        const unsigned iteration_cnt;
        gausian_process<double>* gp_instance;
    public:
        bayes_optimize(unsigned iteration_cnt_v,gausian_process<double>* gp_instance_v):
                        iteration_cnt(iteration_cnt_v),
                        gp_instance(gp_instance_v) {;}
        ~bayes_optimize(){;}
        double optimize(vector<double>& observe_point,vector<double>& observe_value);
        double optimize(vector<double>& observe_point,vector<double>& observe_value,bool is_lower,const double target);
    };

    double linear_predict(vector<double>& x,vector<double>& t,double target);
}

template <typename T>
my_gausian_process::gausian_process<T>::gausian_process(const unsigned predict_cnt_v):
                                        sigma(0.18),
                                        noise(1e-20),
                                        sampling_cnt(0),
                                        predict_cnt(predict_cnt_v)
{
}

template <typename T>
my_gausian_process::gausian_process<T>::gausian_process(vector<T>& a,vector<T>& b,const unsigned predict_cnt_v):
                                                                      curve_fitting_x(a),
                                                                      curve_fitting_t(b),
                                                                      sigma(0.18),
                                                                      noise(1e-20),
                                                                      predict_cnt(predict_cnt_v)
{
    sampling_cnt = curve_fitting_x.size();
}


template <typename T>
my_gausian_process::gausian_process<T>::gausian_process(const gausian_process<T>& a):
gausian_process(a.predict_cnt)
{
    sigma = a.sigma;
    noise = a.noise;
    curve_fitting_x = a.curve_fitting_x;
    curve_fitting_t = a.curve_fitting_t;
    sampling_cnt = a.sampling_cnt;
}



template <typename T>
T my_gausian_process::gausian_process<T>::predict(T target)
{
    //allocate buffer area
    vector<T> buf_m(1);

    //mean of posterior = regression function
    vector<T> outer_buf(sampling_cnt);
    buf_m[0] = target;
    outer(outer_buf,buf_m, curve_fitting_x,sigma);

    vector<T> mul_buf(sampling_cnt);
    multiplication(mul_buf,outer_buf,covariance);
    
    return mul_transpose(mul_buf,curve_fitting_t);
}


template <typename T>
void my_gausian_process::gausian_process<T>::calc_covariance()
{
    // Gram matrix
    unsigned gram_mat_size = sampling_cnt * sampling_cnt;
    vector<T> K(gram_mat_size);
    outer(K,curve_fitting_x, curve_fitting_x,sigma);

    //covariance of marginal
    vector<T> C_N(gram_mat_size);
    for(unsigned i = 0;i < gram_mat_size;++i){
        C_N[i] = K[i] + noise; 
    }

    //create inverse matrix
    covariance.resize(gram_mat_size);
    inv(covariance,C_N,sampling_cnt);
}

template <typename T>
void my_gausian_process::gausian_process<T>::create_predict_point()
{
    //create estimate point
    predict_point.resize(predict_cnt);
    for(unsigned i=0;i < predict_cnt;++i){
        predict_point[i] = 0.01 * i;
    }
}

template <typename T>
void my_gausian_process::gausian_process<T>::predict_meanmatrix(vector<T>& result)
{
    result.resize(predict_cnt);
    //calculate estimate value.
    for(unsigned i=0;i < predict_cnt;++i){
        result[i] =  predict(predict_point[i]);
    }
}

template <typename T>
void my_gausian_process::gausian_process<T>::predict_covfmatrix(vector<T>& result)
{
    result.resize(predict_cnt);

    //calculate estimate value.
    vector<T> buf_m(1);
    for(unsigned i=0;i < predict_cnt;++i){
        //mean of posterior = regression function
        vector<T> outer_buf(sampling_cnt);
        buf_m[0] = predict_point[i];
        outer(outer_buf,buf_m, curve_fitting_x,sigma);

        vector<T> mul_buf(sampling_cnt);
        multiplication(mul_buf,outer_buf,covariance);

        T covf_buf = 1 - mul_transpose(mul_buf,outer_buf);
        result[i] = covf_buf < FLT_EPSILON ? 0 : covf_buf;
    }
}

double my_gausian_process::linear_predict(vector<double>& x,vector<double>& t,double target)
{
    double x0,x1,y0,y1;
    x0 = x1 = y0 = y1 = 0;

    vector<double>::iterator end = x.end();

    vector<double>::iterator hit = find(x.begin(),x.end(),target);
    if(hit != end){
        unsigned index = hit - x.begin();
        return t[index];
    }    

    vector<double>::iterator it = x.begin();
    for(;it!= end;++it){
        if(*it > target){
            x1 = *it;
            x0 = *(it-1);
            unsigned index = it - x.begin();
            y1 = t[index];
            y0 = t[index-1];
            break;
        }
    }
    return y0 + (y1 - y0) * (target - x0) / (x1 - x0);
}

namespace{
    unsigned acq_function(vector<double>& mu,vector<double>& covf){
        const unsigned loop_max = mu.size();
        vector<double> buf(loop_max);
        const double k = 3;
        for(unsigned i=0;i<loop_max;++i){
            buf[i] = mu[i] + k * sqrt(covf[i]);
        }
        unsigned index = std::max_element(buf.begin(), buf.end()) - buf.begin();
        return index;
    }
}

double my_gausian_process::bayes_optimize::optimize(vector<double>& observe_point,vector<double>& observe_value)
{
    vector<double> ob_pt(observe_point);
    vector<double> ob_val(observe_value);

    const unsigned predict_cnt = *(std::max_element(ob_pt.begin(),ob_pt.end()))*100;

    for(unsigned i=0;i < iteration_cnt;++i){
        my_gausian_process::gausian_process< double > gp_buf(ob_pt,ob_val,predict_cnt);
        gp_buf.set_noise(1e-7);
        gp_buf.calc_covariance();
        gp_buf.create_predict_point();
        vector<double> mu(1);
        gp_buf.predict_meanmatrix(mu);

        vector<double> convf(1);
        gp_buf.predict_covfmatrix(convf);

        unsigned index = acq_function(mu,convf);
        double predict_pt_value = gp_buf.get_predict_point_value(index);
        double mean = gp_instance->predict(predict_pt_value);
        ob_pt.push_back(predict_pt_value);
        ob_val.push_back(mean);

        std::cout << index << " ; " << predict_pt_value <<  " ; " << mean<< std::endl;
    }
    unsigned result_index = std::min_element(ob_val.begin(), ob_val.end()) - ob_val.begin();
    std::cout << "RESULT -> index : "<< result_index << ", predict_pt :  " << ob_pt[result_index] << ", value( mean ): " << ob_val[result_index] <<std::endl;
    return ob_pt[result_index];
}

double my_gausian_process::bayes_optimize::optimize(vector<double>& observe_point,vector<double>& observe_value,bool is_lower,const double target)
{
    vector<double> ob_pt(observe_point);
    vector<double> ob_val(observe_value);

    const unsigned predict_cnt = *(std::max_element(ob_pt.begin(),ob_pt.end()))*100;

    for(unsigned i=0;i < iteration_cnt;++i){
        my_gausian_process::gausian_process<double> gp_buf(ob_pt,ob_val,predict_cnt);
        gp_buf.set_noise(1e-7);
        gp_buf.calc_covariance();
        gp_buf.create_predict_point();
        vector<double> mu(1);
        gp_buf.predict_meanmatrix(mu);

        vector<double> convf(1);
        gp_buf.predict_covfmatrix(convf);

        unsigned index = acq_function(mu,convf);
        double predict_pt_value = gp_buf.get_predict_point_value(index);
        double mean = gp_instance->predict(predict_pt_value);
        ob_pt.push_back(predict_pt_value);
        ob_val.push_back(mean);

        std::cout << index << " ; " << predict_pt_value <<  " ; " << mean<< std::endl;
    }
    unsigned size = ob_val.size();
    vector<double> candidate;
    for(unsigned i = 0;i < size;++i){
        double buf = ob_val[i] - target;
        if(is_lower && buf <= 0){
            candidate.push_back(buf);
        }
        if(!is_lower && buf >= 0){
            candidate.push_back(buf);
        }
        std::cout << "kouho ; " << buf << ",(value)" << ob_val[i] << std::endl;
    }
    double buf_element = 0;
    unsigned result_index = 0;
    if(candidate.size() != 0){
        if(is_lower ){
            buf_element = *std::max_element(candidate.begin(), candidate.end());
        }else{
            buf_element = *std::min_element(candidate.begin(), candidate.end());
        }
        result_index = find(ob_val.begin(), ob_val.end(),buf_element + target) - ob_val.begin();
    }else{
        result_index = std::min_element(ob_val.begin(), ob_val.end()) - ob_val.begin();
    }

    std::cout << "RESULT -> index : "<< result_index << ", predict_pt :  " << ob_pt[result_index] << ", value( mean ): " << ob_val[result_index] <<std::endl;
    return ob_pt[result_index];
}

