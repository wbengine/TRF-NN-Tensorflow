w[1:5]  // word 4gram
c[1:5]  // class 4gram
// word skip
w[1]-[1]w[1]	// 3gram ->
w[2]-[1]w[1]    // 4gram ->
w[1]-[1]w[2]
w[1]-[2]w[1]    
w[1]-[1]w[3]    // 5gram ->
w[3]-[1]w[1]
w[2]-[2]w[1]
w[1]-[2]w[2]
w[1]-[3]w[1]
// class skip
c[1]-[1]c[1]	// 3gram ->
c[2]-[1]c[1]    // 4gram ->
c[1]-[1]c[2]
c[1]-[2]c[1]    
c[1]-[1]c[3]    // 5gram ->
c[3]-[1]c[1]
c[2]-[2]c[1]
c[1]-[2]c[2]
c[1]-[3]c[1]

