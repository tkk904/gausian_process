#pragma once

#include <stdlib.h>
#include <math.h>
#include <vector>

using namespace std;

namespace my_gausian_process{

    template <typename T>
    double karnel(T x1,T x2,T sigma)
    {
        // Gaussian kernel
        //Squared Exponential
        return exp(- ((x1-x2) * (x1-x2)) / (2 * sigma * sigma));
        //Ornstein Uhlenbeck
        //return exp(- fabs(x1-x2) / sigma);
    }

    template <typename T>
    void outer(vector<T>& product,vector<T>& a,vector<T>& b,T sigma)
    {
        const unsigned R = a.size();
        const unsigned C = b.size();
      
        for(unsigned r = 0;r < R;++r){
            for(unsigned c = 0;c < C;++c){
                product[r * C + c] = karnel(a[r], b[c],sigma);
            }
        }
    }

    template <typename T>
    void inv(vector<T>& inv_a,vector<T>& a,const unsigned dimention)
    {
        T buf; 
        unsigned i,j,k; 
        unsigned n = dimention;

        //creata unit matrix
        for(i=0;i<n;i++){
            for(j=0;j<n;j++){
                inv_a[i * n + j]=(i==j)?1.0:0.0;
            }
        }
        //hakidashi method
        for(i=0;i<n;i++){
            buf= 1 / a[i * n + i];
            for(j=0;j<n;j++){
                a[ i * n + j]*=buf;
                inv_a[i * n + j]*=buf;
            }
            for(j=0;j<n;j++){
                if(i!=j){
                    buf=a[j * n + i];
                    for(k=0;k<n;k++){
                        a[j * n + k] -= a[i * n + k] * buf;
                        inv_a[j * n + k] -= inv_a[i * n + k] * buf;
                    }
                }
            }
        }
    }

    template <typename T>
    void multiplication(vector<T>& result,vector<T>& a,vector<T>& b)
    {
        const unsigned R = a.size();
        for(unsigned r = 0;r < R;++r){
            for(unsigned c = 0;c < R;++c){
                result[r] += a[c] * b[c * R + r] ;
            }
        }
    }

    template <typename T>
    T mul_transpose(vector<T>& a,vector<T>& b)
    {
        //multiplication transpose matrix
        double result = 0; 
        const unsigned C = b.size();
        for(unsigned c=0;c<C;++c){
            result += a[c] * b[c];
        }
        return result;
    }
}
