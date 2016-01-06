#include "stdafx.h"
#pragma once
#include <vector>
#include <algorithm>
#include <functional>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <strstream>

using namespace std;

int     opterr = 1,             /* if error message should be printed */
optind = 1,             /* index into parent argv vector */
optopt,                 /* character checked for validity */
optreset;               /* reset getopt */
char    *optarg;                /* argument associated with option */

#define BADCH   (int)'?'
#define BADARG  (int)':'
#define EMSG    ""

int
getopt(int nargc, char * const nargv[], const char *ostr)
{
	static char *place = EMSG;              /* option letter processing */
	const char *oli;                        /* option letter list index */

	if (optreset || !*place) {              /* update scanning pointer */
		optreset = 0;
		if (optind >= nargc || *(place = nargv[optind]) != '-') {
			place = EMSG;
			return (-1);
		}
		if (place[1] && *++place == '-') {      /* found "--" */
			++optind;
			place = EMSG;
			return (-1);
		}
	}                                       /* option letter okay? */
	if ((optopt = (int)*place++) == (int)':' ||
		!(oli = strchr(ostr, optopt))) {
		/*
		* if the user didn't specify '-' as an option,
		* assume it means -1.
		*/
		if (optopt == (int)'-')
			return (-1);
		if (!*place)
			++optind;
		if (opterr && *ostr != ':')
			(void)printf("illegal option -- %c\n", optopt);
		return (BADCH);
	}
	if (*++oli != ':') {                    /* don't need argument */
		optarg = NULL;
		if (!*place)
			++optind;
	}
	else {                                  /* need an argument */
		if (*place)                     /* no white space */
			optarg = place;
		else if (nargc <= ++optind) {   /* no arg */
			place = EMSG;
			if (*ostr == ':')
				return (BADARG);
			if (opterr)
				(void)printf("option requires an argument -- %c\n", optopt);
			return (BADCH);
		}
		else                            /* white space */
			optarg = nargv[optind];
		place = EMSG;
		++optind;
	}
	return (optopt);                        /* dump back option letter */
}


int N = 0;                    /* N points(rows) */
int d = -1;                   /* d variables */
float C = 0.05;
float tolerance = 0.001;
float eps = 0.001;
float two_sigma_squared = 2;

vector<float> alph;           /* Lagrange multipliers */
float b;                      /* threshold */
vector<float> w;              /* weight vector: only for linear kernel */

vector<float> error_cache;

struct sparse_binary_vector {
	vector<int> id;
};
struct sparse_vector {
	vector<int> id;
	vector<float> val;
};
typedef vector<float> dense_vector;

bool is_sparse_data = false;
bool is_binary = false;
/* use only one of these */
vector<sparse_binary_vector> sparse_binary_points;
vector<sparse_vector> sparse_points;
vector<dense_vector> dense_points;

vector<int> target;           /* class labels of training data points */
bool is_test_only = false;
bool is_linear_kernel = false;

/* data points with index in [first_test_i .. N)
* will be tested to compute error rate
*/
int first_test_i = 0;

/*
* support vectors are within [0..end_support_i)
*/
int end_support_i = -1;
int takeStep(int i1, int i2);
float(*learned_func)(int) = NULL;
float(*kernel_func)(int, int) = NULL;
float delta_b;
float(*dot_product_func)(int, int) = NULL;
vector<float> precomputed_self_dot_product;
int examineExample(int i1)
{
	float y1, alph1, E1, r1;

	y1 = target[i1];
	alph1 = alph[i1];

	if (alph1 > 0 && alph1 < C)
		E1 = error_cache[i1];
	else
		E1 = learned_func(i1) - y1;

	r1 = y1 * E1;
	if ((r1 < -tolerance && alph1 < C)
		|| (r1 > tolerance && alph1 > 0))
	{
		/* Try i2 by three ways; if successful, then immediately return 1; */

		{
			int k, i2;
			float tmax;

			for (i2 = (-1), tmax = 0, k = 0; k < end_support_i; k++)
				if (alph[k] > 0 && alph[k] < C) {
					float E2, temp;

					cerr << "H1 with k=" << k << endl;

					E2 = error_cache[k];
					temp = fabs(E1 - E2);
					if (temp > tmax)
					{
						tmax = temp;
						i2 = k;
					}
				}

			if (i2 >= 0) {
				if (takeStep(i1, i2)) {
					cerr << "takeStep=1" // << "w=" << w[1] <<" " << w[2] 
						<< " alph=" << alph[0] << " " << alph[1]
						<< " " << alph[2] << " " << alph[3] << " b=" << b
						<< " error_cache=" << error_cache[0] << " " << error_cache[1]
						<< " " << error_cache[2] << " " << error_cache[3] << endl;
					return 1;
				}
			}
		}
		{
			int k, k0;
			int i2;

			for (k0 = 0, //(int) (drand48 () * end_support_i), 
				k = k0; k < end_support_i + k0; k++) {
				i2 = k % end_support_i;
				if (alph[i2] > 0 && alph[i2] < C) {
					cerr << "H2 with k=" << k << endl;
					if (takeStep(i1, i2)) {
						cerr << "takeStep=1" // << "w=" << w[1] <<" " << w[2] 
							<< " alph=" << alph[0] << " " << alph[1]
							<< " " << alph[2] << " " << alph[3] << " b=" << b
							<< " error_cache=" << error_cache[0] << " " << error_cache[1]
							<< " " << error_cache[2] << " " << error_cache[3] << endl;
						return 1;
					}
				}
			}
		}
		{
			int k0, k, i2;

			for (k0 = 0, // (int)(drand48 () * end_support_i), 
				k = k0; k < end_support_i + k0; k++) {
				i2 = k % end_support_i;
				cerr << "H3 with k=" << k << endl;
				if (takeStep(i1, i2)) {
					cerr << "takeStep=1" // << "w=" << w[1] <<" " << w[2] 
						<< " alph=" << alph[0] << " " << alph[1]
						<< " " << alph[2] << " " << alph[3] << " b=" << b
						<< " error_cache=" << error_cache[0] << " " << error_cache[1]
						<< " " << error_cache[2] << " " << error_cache[3] << endl;
					return 1;
				}
			}
		}
	}

	return 0;
}

int takeStep(int i1, int i2) {
	int y1, y2, s;
	float alph1, alph2; /* old_values of alpha_1, alpha_2 */
	float a1, a2;       /* new values of alpha_1, alpha_2 */
	float E1, E2, L, H, k11, k22, k12, eta, Lobj, Hobj;

	cerr << "takeStep(" << i1 << "," << i2 << ")" << endl;

	if (i1 == i2) return 0;

	alph1 = alph[i1];
	y1 = target[i1];
	if (alph1 > 0 && alph1 < C)
		E1 = error_cache[i1];
	else
		E1 = learned_func(i1) - y1;

	alph2 = alph[i2];
	y2 = target[i2];
	if (alph2 > 0 && alph2 < C)
		E2 = error_cache[i2];
	else
		E2 = learned_func(i2) - y2;

	s = y1 * y2;

	cerr << "alph1=" << alph1 << " alph2=" << alph2 << endl;
	cerr << "E1=" << E1 << " E2=" << E2 << " s=" << s << endl;

	if (y1 == y2) {
		float gamma = alph1 + alph2;
		if (gamma > C) {
			L = gamma - C;
			H = C;
		}
		else {
			L = 0;
			H = gamma;
		}
	}
	else {
		float gamma = alph1 - alph2;
		if (gamma > 0) {
			L = 0;
			H = C - gamma;
		}
		else {
			L = -gamma;
			H = C;
		}
	}

	cerr << "L=" << L << " H=" << H << endl;



	if (L == H)
		return 0;



	k11 = kernel_func(i1, i1);
	k12 = kernel_func(i1, i2);
	k22 = kernel_func(i2, i2);
	eta = 2 * k12 - k11 - k22;



	if (eta < 0) {
		a2 = alph2 + y2 * (E2 - E1) / eta;
		if (a2 < L)
			a2 = L;
		else if (a2 > H)
			a2 = H;
	}
	else {


		{
			float c1 = eta / 2;
			float c2 = y2 * (E1 - E2) - eta * alph2;
			Lobj = c1 * L * L + c2 * L;
			Hobj = c1 * H * H + c2 * H;
		}


		if (Lobj > Hobj + eps)
			a2 = L;
		else if (Lobj < Hobj - eps)
			a2 = H;
		else
			a2 = alph2;
	}

	cerr << "a2=" << a2 << " alph2=" << alph2 << endl;

	if (fabs(a2 - alph2) < eps*(a2 + alph2 + eps))
		return 0;

	a1 = alph1 - s * (a2 - alph2);
	if (a1 < 0) {
		a2 += s * a1;
		a1 = 0;
	}
	else if (a1 > C) {
		float t = a1 - C;
		a2 += s * t;
		a1 = C;
	}



	{
		float b1, b2, bnew;

		if (a1 > 0 && a1 < C)
			bnew = b + E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12;
		else {
			if (a2 > 0 && a2 < C)
				bnew = b + E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22;
			else {
				b1 = b + E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12;
				b2 = b + E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22;
				bnew = (b1 + b2) / 2;
			}
		}

		delta_b = bnew - b;
		b = bnew;
	}





	if (is_linear_kernel) {
		float t1 = y1 * (a1 - alph1);
		float t2 = y2 * (a2 - alph2);

		if (is_sparse_data && is_binary) {
			int p1, num1, p2, num2;

			num1 = sparse_binary_points[i1].id.size();
			for (p1 = 0; p1 < num1; p1++)
				w[sparse_binary_points[i1].id[p1]] += t1;

			num2 = sparse_binary_points[i2].id.size();
			for (p2 = 0; p2 < num2; p2++)
				w[sparse_binary_points[i2].id[p2]] += t2;
		}
		else if (is_sparse_data && !is_binary) {
			int p1, num1, p2, num2;

			num1 = sparse_points[i1].id.size();
			for (p1 = 0; p1 < num1; p1++)
				w[sparse_points[i1].id[p1]] +=
				t1 * sparse_points[i1].val[p1];

			num2 = sparse_points[i2].id.size();
			for (p2 = 0; p2 < num2; p2++)
				w[sparse_points[i2].id[p2]] +=
				t2 * sparse_points[i2].val[p2];
		}
		else
			for (int i = 0; i < d; i++)
				w[i] += dense_points[i1][i] * t1 + dense_points[i2][i] * t2;
	}




	{
		float t1 = y1 * (a1 - alph1);
		float t2 = y2 * (a2 - alph2);

		cerr << "t1=" << t1 << " t2=" << t2 << " delta_b=" << delta_b << endl;

		for (int i = 0; i < end_support_i; i++)
			if (0 < alph[i] && alph[i] < C)
				error_cache[i] += t1 * kernel_func(i1, i) + t2 * kernel_func(i2, i)
				- delta_b;
		error_cache[i1] = 0.;
		error_cache[i2] = 0.;
	}



	alph[i1] = a1;  /* Store a1 in the alpha array.*/
	alph[i2] = a2;  /* Store a2 in the alpha array.*/

	return 1;
}

float learned_func_linear_sparse_binary(int k) {
	float s = 0.;

	for (int i = 0; i < sparse_binary_points[k].id.size(); i++) {
		s += w[sparse_binary_points[k].id[i]];
	}

	s -= b;
	return s;
}

float learned_func_linear_sparse_nonbinary(int k) {
	float s = 0.;

	for (int i = 0; i < sparse_points[k].id.size(); i++)
	{
		int j = sparse_points[k].id[i];
		float v = sparse_points[k].val[i];
		s += w[j] * v;
	}
	s -= b;
	return s;
}

float learned_func_linear_dense(int k) {
	float s = 0.;

	cout << "results are: " << endl;

	for (int i = 0; i < d; i++) {
		s += w[i] * dense_points[k][i];
		cout << "Weights: " << w[i] << " and result: " << w[i] *dense_points[k][i] << endl;

	}

	s -= b;
	return s;
}

float learned_func_nonlinear(int k) {
	float s = 0.;
	for (int i = 0; i < end_support_i; i++)
		if (alph[i] > 0)
			s += alph[i] * target[i] * kernel_func(i, k);
	s -= b;
	return s;
}


float dot_product_sparse_binary(int i1, int i2)
{
	int p1 = 0, p2 = 0, dot = 0;
	int num1 = sparse_binary_points[i1].id.size();
	int num2 = sparse_binary_points[i2].id.size();

	while (p1 < num1 && p2 < num2) {
		int a1 = sparse_binary_points[i1].id[p1];
		int a2 = sparse_binary_points[i2].id[p2];
		if (a1 == a2) {
			dot++;
			p1++;
			p2++;
		}
		else if (a1 > a2)
			p2++;
		else
			p1++;
	}
	return (float)dot;
}


float dot_product_sparse_nonbinary(int i1, int i2)
{
	int p1 = 0, p2 = 0;
	float dot = 0.;
	int num1 = sparse_points[i1].id.size();
	int num2 = sparse_points[i2].id.size();

	while (p1 < num1 && p2 < num2) {
		int a1 = sparse_points[i1].id[p1];
		int a2 = sparse_points[i2].id[p2];
		if (a1 == a2) {
			dot += sparse_points[i1].val[p1] * sparse_points[i2].val[p2];
			p1++;
			p2++;
		}
		else if (a1 > a2)
			p2++;
		else
			p1++;
	}
	return (float)dot;
}


float dot_product_dense(int i1, int i2)
{
	float dot = 0.;
	for (int i = 0; i < d; i++)
		dot += dense_points[i1][i] * dense_points[i2][i];

	return dot;
}



float rbf_kernel(int i1, int i2)
{
	float s = dot_product_func(i1, i2);
	s *= -2;
	s += precomputed_self_dot_product[i1] + precomputed_self_dot_product[i2];
	return exp(-s / two_sigma_squared);
}


int read_data(istream& is)
{
	string s;
	int n_lines;

	for (n_lines = 0; getline(is, s, '\n'); n_lines++) {
		istrstream line(s.c_str());
		vector<float> v;
		float t;
		while (line >> t)
			v.push_back(t);
		target.push_back(v.back());
		v.pop_back();
		int n = v.size();
		if (is_sparse_data && is_binary) {
			sparse_binary_vector x;
			for (int i = 0; i < n; i++) {
				if (v[i] < 1 || v[i] > d) {
					cerr << "error: line " << n_lines + 1
						<< ": attribute index " << int(v[i]) << " out of range." << endl;
					exit(1);
				}
				x.id.push_back(int(v[i]) - 1);
			}
			sparse_binary_points.push_back(x);
		}
		else if (is_sparse_data && !is_binary) {
			sparse_vector x;
			for (int i = 0; i < n; i += 2) {
				if (v[i] < 1 || v[i] > d) {
					cerr << "data file error: line " << n_lines + 1
						<< ": attribute index " << int(v[i]) << " out of range."
						<< endl;
					exit(1);
				}
				x.id.push_back(int(v[i]) - 1);
				x.val.push_back(v[i + 1]);
			}
			sparse_points.push_back(x);
		}
		else {
			if (v.size() != d) {
				cerr << "data file error: line " << n_lines + 1
					<< " has " << v.size() << " attributes; should be d=" << d
					<< endl;
				exit(1);
			}
			dense_points.push_back(v);
		}
	}
	return n_lines;
}


void write_svm(ostream& os) {
	os << d << endl;
	os << is_sparse_data << endl;
	os << is_binary << endl;
	os << is_linear_kernel << endl;
	os << b << endl;
	if (is_linear_kernel) {
		for (int i = 0; i < d; i++)
			os << w[i] << endl;
	}
	else {
		os << two_sigma_squared << endl;
		int n_support_vectors = 0;
		for (int i = 0; i < end_support_i; i++)
			if (alph[i] > 0)
				n_support_vectors++;
		os << n_support_vectors << endl;
		for (int i = 0; i < end_support_i; i++)
			if (alph[i] > 0)
				os << alph[i] << endl;
		for (int i = 0; i < end_support_i; i++)
			if (alph[i] > 0) {
				if (is_sparse_data && is_binary) {
					for (int j = 0; j < sparse_binary_points[i].id.size(); j++)
						os << (sparse_binary_points[i].id[j] + 1) << ' ';
				}
				else if (is_sparse_data && !is_binary) {
					for (int j = 0; j < sparse_points[i].id.size(); j++)
						os << (sparse_points[i].id[j] + 1) << ' '
						<< sparse_points[i].val[j] << ' ';
				}
				else {
					for (int j = 0; j < d; j++)
						os << dense_points[i][j] << ' ';
				}
				os << target[i];
				os << endl;
			}
	}
}


int read_svm(istream& is) {
	is >> d;
	is >> is_sparse_data;
	is >> is_binary;
	is >> is_linear_kernel;
	is >> b;
	if (is_linear_kernel) {
		w.resize(d);
		for (int i = 0; i < d; i++)
			is >> w[i];
	}
	else {
		is >> two_sigma_squared;
		int n_support_vectors;
		is >> n_support_vectors;
		alph.resize(n_support_vectors, 0.);
		for (int i = 0; i < n_support_vectors; i++)
			is >> alph[i];
		string dummy_line_to_skip_newline;
		getline(is, dummy_line_to_skip_newline, '\n');
		return read_data(is);
	}
	return 0;
}


float
error_rate()
{
	int n_total = 0;
	int n_error = 0;
	for (int i = first_test_i; i < N; i++) {
		if (learned_func(i) > 0 != target[i] > 0)
			n_error++;
		n_total++;
	}
	return float(n_error) / float(n_total);
}

int smomain(int argc, char *argv[]) {
	char *data_file_name = "svm.data";
	char *svm_file_name = "svm.model";
	char *output_file_name = "svm.output";
	int numChanged;
	int examineAll;
	{
		extern char *optarg;
		extern int optind;
		int c;
		int errflg = 0;

		while ((c = getopt(argc, argv, "n:d:c:t:e:p:f:m:o:r:lsba")) != EOF)
			switch (c)
			{
			case 'n':
				N = atoi(optarg);
				break;
			case 'd':
				d = atoi(optarg);
				break;
			case 'c':
				C = atof(optarg);
				break;
			case 't':
				tolerance = atof(optarg);
				break;
			case 'e':
				eps = atof(optarg);
				break;
			case 'p':
				two_sigma_squared = atof(optarg);
				break;
			case 'f':
				data_file_name = optarg;
				break;
			case 'm':
				svm_file_name = optarg;
				break;
			case 'o':
				output_file_name = optarg;
				break;
			case 'r':
				//srand48(atoi(optarg));
				break;
			case 'l':
				is_linear_kernel = true;
				break;
			case 's':
				is_sparse_data = true;
				break;
			case 'b':
				is_binary = true;
				break;
			case 'a':
				is_test_only = true;
				break;
			case '?':
				errflg++;
			}

		if (errflg || optind < argc)
		{
			cerr << "usage: " << argv[0] << " " <<
				"-f  data_file_name\n"
				"-m  svm_file_name\n"
				"-o  output_file_name\n"
				"-n  N\n"
				"-d  d\n"
				"-c  C\n"
				"-t  tolerance\n"
				"-e  epsilon\n"
				"-p  two_sigma_squared\n"
				"-r  random_seed\n"
				"-l  (is_linear_kernel)\n"
				"-s  (is_sparse_data)\n"
				"-b  (is_binary)\n"
				"-a  (is_test_only)\n"
				;
			return 0;
			//exit(2);
		}
	}




	{
		int n;
		if (is_test_only) {
			ifstream svm_file(svm_file_name);
			end_support_i = first_test_i = n = read_svm(svm_file);
			N += n;
		}
		if (N > 0) {
			target.reserve(N);
			if (is_sparse_data && is_binary)
				sparse_binary_points.reserve(N);
			else if (is_sparse_data && !is_binary)
				sparse_points.reserve(N);
			else
				dense_points.reserve(N);
		}
		ifstream data_file(data_file_name);
		n = read_data(data_file);
		if (is_test_only) {
			N = first_test_i + n;
		}
		else {
			N = n;
			first_test_i = 0;
			end_support_i = N;
		}
	}



	if (!is_test_only) {
		alph.resize(end_support_i, 0.);

		/* initialize threshold to zero */
		b = 0.;

		/* E_i = u_i - y_i = 0 - y_i = -y_i */
		error_cache.resize(N);

		if (is_linear_kernel)
			w.resize(d, 0.);
	}




	if (is_linear_kernel && is_sparse_data && is_binary)
		learned_func = learned_func_linear_sparse_binary;
	if (is_linear_kernel && is_sparse_data && !is_binary)
		learned_func = learned_func_linear_sparse_nonbinary;
	if (is_linear_kernel && !is_sparse_data)
		learned_func = learned_func_linear_dense;
	if (!is_linear_kernel)
		learned_func = learned_func_nonlinear;



	if (is_sparse_data && is_binary)
		dot_product_func = dot_product_sparse_binary;
	if (is_sparse_data && !is_binary)
		dot_product_func = dot_product_sparse_nonbinary;
	if (!is_sparse_data)
		dot_product_func = dot_product_dense;



	if (is_linear_kernel)
		kernel_func = dot_product_func;
	if (!is_linear_kernel)
		kernel_func = rbf_kernel;



	if (!is_linear_kernel) {
		precomputed_self_dot_product.resize(N);
		for (int i = 0; i < N; i++)
			precomputed_self_dot_product[i] = dot_product_func(i, i);
	}




	if (!is_test_only) {
		numChanged = 0;
		examineAll = 1;
		while (numChanged > 0 || examineAll) {
			numChanged = 0;
			if (examineAll) {
				for (int k = 0; k < N; k++)
					numChanged += examineExample(k);
			}
			else {
				for (int k = 0; k < N; k++)
					if (alph[k] != 0 && alph[k] != C)
						numChanged += examineExample(k);
			}
			if (examineAll == 1)
				examineAll = 0;
			else if (numChanged == 0)
				examineAll = 1;

			//cerr << error_rate() << endl;



			/* L_D */
			{
#if 0
				float s = 0.;
				for (int i = 0; i < N; i++)
					s += alph[i];
				float t = 0.;
				for (int i = 0; i < N; i++)
					for (int j = 0; j < N; j++)
						t += alph[i] * alph[j] * target[i] * target[j] * kernel_func(i, j);
				cerr << "Objective function=" << (s - t / 2.) << endl;
				for (int i = 0; i < N; i++)
					if (alph[i] < 0)
						cerr << "alph[" << i << "]=" << alph[i] << " < 0" << endl;
				s = 0.;
				for (int i = 0; i < N; i++)
					s += alph[i] * target[i];
				cerr << "s=" << s << endl;
				cerr << "error_rate=" << error_rate() << '\t';
#endif
				int non_bound_support = 0;
				int bound_support = 0;
				for (int i = 0; i < N; i++)
					if (alph[i] > 0) {
						if (alph[i] < C)
							non_bound_support++;
						else
							bound_support++;
					}
				cerr << "non_bound=" << non_bound_support << '\t';
				cerr << "bound_support=" << bound_support << endl;
			}



		}


		{
			if (!is_test_only && svm_file_name != NULL) {
				ofstream svm_file(svm_file_name);
				write_svm(svm_file);
			}
		}


		cerr << "threshold=" << b << endl;
	}
	cout << error_rate() << endl;


	{
		ofstream output_file(output_file_name);
		for (int i = first_test_i; i < N; i++)
			output_file << learned_func(i) << endl;
	}


}