## Instalacja

	Folder randomWalk zawiera projekt c++ NeatBeans IDE.
	Jest kompilowany za pomocą g++ w wersji 6.1.1.
	Folder zawiera także automatycznie generowany przez IDE makefile,
	więc kompilacja nie powinna sprawiać problemu również poza środowiskiem.
	
	Folder randomWalk zawiera także projekt c++ Visual Studio 2013 (Toolset v12)
	Wystarczy otworzyć plik .sln oraz skompilować.
	
	Program może zostać wywołany z użyciem parametrów:
		parametry:
		1: ścieżka do testu
		2: ilość wywołań
		3: startowy Rect
		
	W przeciwnym razie zostaną wywołane domyślne opcje, zdefiniowane w pliku main.cpp
	
## Fragmenty kodu które mamy zamiar zrównoleglać

	Cześć kodu jaką chlelibyśmy zrównoleglać to metody:
	- bool Tree::checkCollisions(Rect const& r, const Rect &ignore) (plik quadTree.cpp linia 153),
	- bool Tree::checkCollisons(point p, Rect& r) (plik quadTree.cpp linia 244),
	- Rect Tree::drawBiggestSquareAtPoint(point p) (plik quadTree.cpp linia 268).
	Prosimy Pana o sugestie, z góry bardzo dziękujemy.