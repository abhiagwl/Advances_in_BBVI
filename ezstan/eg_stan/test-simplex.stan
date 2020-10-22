data {
  int<lower=2> V;               // num words
  int<lower=1> N;               // total word instances
  vector<lower=0>[V] beta;      // word prior
  int<lower=1,upper=V> doc[N];  // doc ID for word n
}
parameters {
  simplex[V] phi;     // word dist
}
model {
  phi ~ dirichlet(beta);     // prior
  for (n in 1:N) {
    increment_log_prob(log( phi[doc[n]] ));
  }
}
