#include <stdlib.h>
#include <iostream>
#include "gp_estimate.h"

using namespace std;

int main(int argc,char *argv[])
{
//    vector<double> curve_fitting_x {0.000000,0.111111,0.222222,0.333333,0.444444,0.555556,0.666667,0.777778,0.888889,1.000000};
//    vector<double> curve_fitting_t {0.349486,0.830839,1.007332,0.971507,0.133066,0.166823,-0.848307,-0.445686,-0.563567,0.261502};
    vector<double> curve_fitting_x {0.1,0.13,0.2,0.25,0.3,0.4,0.5,0.6,0.75,0.8,1.0};
    vector<double> curve_fitting_t {30,60,64,70,77,80,82,90,100,120,150 };
//    vector<double> curve_fitting_x {0.05,0.1,0.15,0.22,0.25,0.75,0.8};
//    vector<double> curve_fitting_t {30,30,60,66,77,109,120 };
    const unsigned predict_cnt = *(std::max_element(curve_fitting_x.begin(),curve_fitting_x.end()))*150;
    my_gausian_process::gausian_process<double> gp_instance(curve_fitting_x,curve_fitting_t,predict_cnt);

    gp_instance.calc_covariance();

    //create estimate point
    const unsigned estimate_cnt = predict_cnt;
    vector<double> x(estimate_cnt);
    for(unsigned i=0;i < estimate_cnt;++i){
        x[i] = 0.01 * i;
    }

    //calculate estimate value.
    for(unsigned i=0;i < estimate_cnt;++i){
        std::cout << gp_instance.predict(x[i]) << std::endl;
    }
    std::cout << std::endl;

    // test : gausian_process get mean & covf
    gp_instance.set_noise(1e-20);
    gp_instance.calc_covariance();

    gp_instance.create_predict_point();
    vector<double> predict_mean_result(1);
    gp_instance.predict_meanmatrix(predict_mean_result);

    vector<double> predict_covf_result(1);
    gp_instance.predict_covfmatrix(predict_covf_result);

    vector<double>::iterator mean_it;
    vector<double>::iterator covf_it = predict_covf_result.begin();
/*
    std::cout << " mean " <<" : "<< " covf " << std::endl;
    for(mean_it = predict_mean_result.begin();mean_it != predict_mean_result.end();++mean_it,++covf_it){
        std::cout << *mean_it <<" : "<< *covf_it << std::endl;
    }
*/
    std::cout << std::endl;

    // test : bayes_opt
    my_gausian_process::bayes_optimize op(20,&gp_instance);
    op.optimize(curve_fitting_x,curve_fitting_t);

    my_gausian_process::bayes_optimize op2(20,&gp_instance);
//    op2.optimize(curve_fitting_x,curve_fitting_t,false,0.878);


    vector<double> linear_x {0,0.04,0.1,0.2,0.5,1.0,1.2};
    vector<double> linear_t {0,1.4,1.42,1,0.5,0,0};

    
    for(unsigned i=0;i < estimate_cnt;++i){
//        std::cout << gausian_process::linear_predict(linear_x,linear_t,x[i]) << std::endl;
    }
    std::cout << std::endl;


    return 0;
}
