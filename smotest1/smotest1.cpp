// smotest1.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <string>
#include "smo.h"

using namespace std;

int main(int argc, char *argv[])
{
	smomain(argc, argv);
	cout << "Completed! Press enter to exit." << endl;
	string s;
	cin >> s;
    return 0;
}