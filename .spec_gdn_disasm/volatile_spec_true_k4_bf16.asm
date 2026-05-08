L0:
(W)     and (1|M0)               r127.0<1>:ud  r0.0<0;1,0>:ud    0xFFFFFFC0:ud             
(W)     add (1|M0)               r127.0<1>:ud  r127.0<0;1,0>:ud  0x0:ud              {I@1}
(W)     send.ugm (1|M0)          r2       r127  null:0  0xFF000000            0x6219D500           {A@1,$0} // wr:1+0, rd:1; load.ugm.d32x16t.a32.ca.cc.bti[255]
(W)     send.ugm (1|M0)          r3       r127  null:0  0xFF040000            0x6219C500           {$1} // wr:1+0, rd:1; load.ugm.d32x8t.a32.ca.cc.bti[255][A+0x40]
(W)     mov (16|M0)              r21.0<1>:ud   r0.0<1;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               r4.0<1>:f     9.18355e-41:f                              
(W)     and (1|M0)               r1.9<1>:ud    r21.5<0;1,0>:ud   0xFFFFFC00:ud              {I@1}
(W)     or (1|M0)                cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x400004C0:ud              {A@1}
(W)     mov (8|M0)               r48.0<1>:w    0x76543210:v                               {A@1}
(W)     cmp (32|M0)   (eq)f3.0   null<1>:d     r3.1<0;1,0>:d     0:w               {$1.dst}
(W)     add (8|M0)               r48.8<1>:w    r48.0<1;1,0>:w    8:w               {I@2}
(W)     mov (1|M0)               r3.15<1>:d    r21.1<0;1,0>:d                  
(W)     add (16|M0)              r48.16<1>:w   r48.0<1;1,0>:w    16:w               {I@2}
(W)     mov (1|M0)               r4.4<2>:b     r21.8<0;1,0>:b                  
(W&~f3.0) jmpi                               L264                                
L232:
(W)     mov (1|M0)               r3.8<1>:d     -1:w                              
(W)     jmpi                                 L816                                
L264:
(W)     asr (1|M0)               r1.15<1>:d    r3.1<0;1,0>:d     31:w              
(W)     asr (1|M0)               r1.14<1>:d    r3.3<0;1,0>:d     31:w              
(W)     add (1|M0)               r1.10<1>:d    r1.15<0;1,0>:d    r3.1<0;1,0>:d    {I@2}
(W)     xor (1|M0)               r1.11<1>:d    r1.10<0;1,0>:d    r1.15<0;1,0>:d   {I@1}
(W)     add (1|M0)               r1.10<1>:d    r1.14<0;1,0>:d    r3.3<0;1,0>:d   
(W)     xor (1|M0)               r3.9<1>:d     r1.10<0;1,0>:d    r1.14<0;1,0>:d   {I@1}
(W)     xor (1|M0)               cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x30:uw              {A@1}
(W)     mov (1|M0)               r3.7<1>:f     r1.11<0;1,0>:ud                  {A@1}
(W)     mov (1|M0)               r3.6<1>:f     r3.9<0;1,0>:ud                   {I@2}
(W)     mov (1|M0)               r1.10<1>:ud   r3.7<0;1,0>:f                    {F@2}
(W)     math.inv (1|M0)          r3.10<1>:f    r3.7<0;1,0>:f                   
(W)     add (1|M0)               r3.12<1>:d    r1.11<0;1,0>:d    -r1.10<0;1,0>:d  {I@1}
(W)     mov (1|M0)               r1.10<1>:f    0xB4C00000:f                               {I@1}
(W)     mov (1|M0)               r1.12<1>:f    r3.12<0;1,0>:ud                 
(W)     mad (1|M0)               r3.14<1>:f    r3.10<0;0>:f      r1.10<0;0>:f      r3.10<0>:f       {A@1}
(W)     mov (1|M0)               r1.10<1>:ud   r3.6<0;1,0>:f                    {F@1}
(W)     mul (1|M0)               r3.10<1>:f    r3.6<0;1,0>:f     r3.14<0;1,0>:f  
(W)     add (1|M0)               r3.13<1>:d    r3.9<0;1,0>:d     -r1.10<0;1,0>:d  {I@1}
(W)     mov (1|M0)               r3.11<1>:ud   r3.10<0;1,0>:f                   {F@1}
(W)     mov (1|M0)               r1.13<1>:f    r3.13<0;1,0>:ud                  {I@2}
(W)     mov (1|M0)               r3.10<1>:f    r3.11<0;1,0>:ud                  {I@1}
(W)     mad (1|M0)               r3.6<1>:f     r3.6<0;0>:f       r3.10<0;0>:f      -r3.7<0>:f       {F@1}
(W)     mad (1|M0)               r1.10<1>:f    r1.13<0;0>:f      r3.10<0;0>:f      -r1.12<0>:f     
(W)     add (1|M0)               r1.10<1>:f    r3.6<0;1,0>:f     r1.10<0;1,0>:f   {F@1}
(W)     mul (1|M0)               r3.6<1>:f     r3.14<0;1,0>:f    r1.10<0;1,0>:f   {F@1}
(W)     xor (1|M0)               cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x30:uw              {A@1}
(W)     mov (1|M0)               r1.10<1>:ud   r3.6<0;1,0>:f                    {A@1}
(W)     xor (1|M0)               r3.7<1>:d     r1.15<0;1,0>:d    r1.14<0;1,0>:d  
(W)     add (1|M0)               r3.6<1>:d     r1.10<0;1,0>:d    r3.11<0;1,0>:d   {I@2}
(W)     mul (1|M0)               acc0.0<1>:d   r3.6<0;1,0>:d     r1.22<0;1,0>:uw  {I@1}
(W)     macl (1|M0)              r5.0<1>:d     r3.6<0;1,0>:d     r1.11<0;1,0>:d   {Compacted}
(W)     add (1|M0)               r1.10<1>:d    r3.9<0;1,0>:d     -r5.0<0;1,0>:d   {I@1}
(W)     cmp (1|M0)    (ge)f0.0   r1.10<1>:ud   r1.10<0;1,0>:ud   r1.11<0;1,0>:ud  {I@1}
(W)     add3 (1|M0)              r1.10<1>:d    r3.6<0;0>:d       r3.7<0;0>:d       -r1.10<0>:d      {I@1}
(W)     bfn.(s0^s1^s2) (1|M0)    r3.8<1>:ud    r1.10<0;0>:ud     r1.15<0;0>:ud     r1.14<0>:ud      {I@1}
L816:
(W)     mov (1|M0)               r1.10<1>:d    r4.4<0;1,0>:ub                  
(W)     shl (1|M0)               r1.11<1>:d    r3.15<0;1,0>:d    5:w              
(W)     shl (1|M0)               r1.10<1>:d    r1.10<0;1,0>:d    2:w               {I@2}
(W)     add (1|M0)               r3.6<1>:d     r1.11<0;1,0>:d    r1.10<0;1,0>:d   {I@1}
(W)     cmp (32|M0)   (lt)f2.0   null<1>:d     r3.6<0;1,0>:d     r3.4<0;1,0>:d    {I@1}
(W&~f2.0) jmpi                               L14048                                
L912:
(W)     mov (1|M0)               r3.7<1>:d     r21.6<0;1,0>:d                  
(W)     shl (1|M0)               r1.7<1>:q     r21.7<0;1,0>:ud   2:w              
(W)     shl (1|M0)               r3.6<1>:q     r3.7<0;1,0>:ud    2:w               {I@2}
(W)     add (1|M0)               r6.0<1>:q     r1.7<0;1,0>:q     r2.7<0;1,0>:q    {Compacted,@2,$0.dst}
(W)     add (1|M0)               r10.0<1>:q    r3.6<0;1,0>:q     r2.2<0;1,0>:q    {Compacted,I@2}
(W)     send.ugm (1|M0)          r7       r6  null:0  0x0            0x02108580           {I@2,$2} // wr:1+0, rd:1; load.ugm.d32x1t.a64
        sync.nop                             null                             {Compacted,$2.src}
(W)     send.ugm (1|M0)          r6       r10  null:0  0x0            0x02108580           {I@1,$3} // wr:1+0, rd:1; load.ugm.d32x1t.a64
(W)     shl (1|M0)               r2.7<1>:q     r3.7<0;1,0>:ud    1:w              
(W)     mul (1|M0)               acc0.0<1>:d   r3.7<0;1,0>:d     r3.4<0;1,0>:uw  
(W)     add (1|M0)               r8.0<1>:q     r2.7<0;1,0>:q     r2.3<0;1,0>:q    {Compacted,I@2}
(W)     macl (1|M0)              r5.0<1>:d     r3.7<0;1,0>:d     r3.2<0;1,0>:d   
(W)     send.ugm (1|M0)          r8       r8  null:0  0x0            0x04100B80           {I@2,$4} // wr:2+0, rd:1; load.ugm.d16u32.a64
(W)     mul (1|M0)               acc0.0<1>:d   r5.0<0;1,0>:d     r3.8<0;1,0>:uw   {I@1}
(W)     macl (1|M0)              r5.0<1>:d     r5.0<0;1,0>:d     r3.4<0;1,0>:d    {Compacted}
(W)     mul (1|M0)               acc0.0<1>:d   r21.7<0;1,0>:d    r3.10<0;1,0>:uw 
(W)     mov (1|M0)               r2.7<1>:d     r5.0<0;1,0>:d                    {I@2}
(W)     macl (1|M0)              r5.0<1>:d     r21.7<0;1,0>:d    r3.5<0;1,0>:d   
        mov (32|M0)              r108.0<1>:d   r48.0<1;1,0>:uw                 
(W)     mov (1|M0)               r2.14<1>:d    r5.0<0;1,0>:d                    {I@2}
        shl (32|M0)              r22.0<1>:d    r108.0<1;1,0>:d   2:w               {Compacted,I@2}
(W)     mov (1|M0)               r3.10<1>:ud   0xBF317200:ud                             
(W)     mov (1|M0)               r3.9<1>:f     0xB5BFBE8E:f                              
(W)     cmp (32|M0)   (lt)f1.0   null<1>:d     r7.0<0;1,0>:d     1:w               {$2.dst}
(W)     mul (1|M0)               r2.4<1>:f     r6.0<0;1,0>:f     1.442695e+00:f               {$3.dst}
(W)     cmp (1|M0)    (lt)f3.0   null<1>:f     r6.0<0;1,0>:f     -105.0:f              
(W)     rndz (1|M0)              r5.0<1>:f     r2.4<0;1,0>:f                    {Compacted,A@2}
(W)     mov (1|M0)               r2.4<1>:f     0xBF317200:f                              
(W)     cmp (1|M0)    (gt)f2.0   null<1>:f     r6.0<0;1,0>:f     105.0:f              
(W)     mad (1|M0)               r2.5<1>:f     r6.0<0;0>:f       r2.4<0;0>:f       r5.0<0>:f        {F@2}
(W)     mov (1|M0)               r2.4<1>:f     0xB5BFBE8E:f                              
(W)     math.exp (1|M0)          r6.0<1>:f     r5.0<0;1,0>:f                    {F@2}
(W)     mad (1|M0)               r2.4<1>:f     r2.5<0;0>:f       r2.4<0;0>:f       r5.0<0>:f        {F@1}
(W)     shl (1|M0)               r4.14<1>:ud   r8.0<0;1,0>:uw    0x10:uw              {$4.dst}
(W)     mul (1|M0)               r5.1<1>:f     r2.4<0;1,0>:f     1.442695e+00:f               {F@1}
(W)     math.exp (1|M0)          r6.1<1>:f     r5.1<0;1,0>:f                    {F@1}
(W)     mul (1|M0)               r2.4<1>:f     r6.0<0;1,0>:f     r6.1<0;1,0>:f    {Compacted,M@1}
(W&~f3.0) sel (1|M0)             r2.4<1>:f     r2.4<0;1,0>:f     0.0:f               {F@1}
(W&~f2.0) sel (1|M0)             r4.10<1>:f    r2.4<0;1,0>:f     inf:f               {F@1}
(W&f1.0) jmpi                                L1752                                
L1488:
(W)     add3 (1|M0)              r2.4<1>:d     r2.14<0;0>:d      r7.0<0;0>:d       -1:w               {F@1}
(W)     shl (1|M0)               r2.2<1>:q     r2.4<0;1,0>:d     2:w               {I@1}
(W)     add (1|M0)               r6.0<1>:q     r2.2<0;1,0>:q     r2.6<0;1,0>:q    {I@1}
(W)     asr (1|M0)               r2.4<1>:d     r3.0<0;1,0>:d     31:w               {Compacted}
(W)     send.ugm (1|M0)          r5       r6  null:0  0x0            0x02108580           {I@2,$5} // wr:1+0, rd:1; load.ugm.d32x1t.a64
(W)     mul (1|M0)               acc0.0<1>:ud  r5.0<0;1,0>:ud    r3.0<0;1,0>:uw   {$5.dst}
(W)     macl (1|M0)              r8.0<1>:ud    r5.0<0;1,0>:ud    r3.0<0;1,0>:ud   {Compacted}
(W)     mul (1|M0)               acc0.0<1>:ud  r5.0<0;1,0>:ud    r3.0<0;1,0>:uw  
(W)     asr (1|M0)               r2.5<1>:d     r5.0<0;1,0>:d     31:w              
(W)     mach (1|M0)              r7.0<1>:d     r5.0<0;1,0>:ud    r3.0<0;1,0>:ud  
(W)     mul (1|M0)               acc0.0<1>:d   r5.0<0;1,0>:ud    r2.8<0;1,0>:uw   {I@6}
(W)     cmp (32|M0)   (gt)f1.0   null<1>:d     r5.0<0;1,0>:d     0:w              
(W)     macl (1|M0)              r6.0<1>:d     r5.0<0;1,0>:ud    r2.4<0;1,0>:d   
(W)     mul (1|M0)               acc0.0<1>:d   r3.0<0;1,0>:ud    r2.10<0;1,0>:uw  {I@5}
(W)     add (1|M0)               r7.0<1>:d     r7.0<0;1,0>:d     r6.0<0;1,0>:d    {Compacted,I@2}
(W)     macl (1|M0)              r6.0<1>:d     r3.0<0;1,0>:ud    r2.5<0;1,0>:d   
(W)     add (1|M0)               r3.11<1>:d    r7.0<0;1,0>:d     r6.0<0;1,0>:d    {I@1}
(W&f1.0) jmpi                                L1896                                
L1752:
        mov (32|M0)              r56.0<1>:ud   0x0:ud                              {Compacted}
        mov (32|M0)              r58.0<1>:ud   0x0:ud                              {Compacted}
        mov (32|M0)              r60.0<1>:ud   0x0:ud                              {Compacted}
        mov (32|M0)              r62.0<1>:ud   0x0:ud                              {Compacted}
        mov (32|M0)              r64.0<1>:ud   0x0:ud                              {Compacted}
        mov (32|M0)              r66.0<1>:ud   0x0:ud                              {Compacted}
        mov (32|M0)              r68.0<1>:ud   0x0:ud                              {Compacted}
        mov (32|M0)              r70.0<1>:ud   0x0:ud                              {Compacted}
        mov (32|M0)              r54.0<1>:f    0.0:f                               {Compacted}
        mov (32|M0)              r50.0<1>:f    0.0:f                               {Compacted}
        mov (32|M0)              r34.0<1>:f    0.0:f                               {Compacted}
        mov (32|M0)              r32.0<1>:f    0.0:f                               {Compacted}
        mov (32|M0)              r30.0<1>:f    0.0:f                               {Compacted}
        mov (32|M0)              r28.0<1>:f    0.0:f                               {Compacted}
        mov (32|M0)              r26.0<1>:f    0.0:f                               {Compacted}
        mov (32|M0)              r24.0<1>:f    0.0:f                               {Compacted}
(W)     jmpi                                 L4736                                
L1896:
(W)     or (1|M0)                r2.6<1>:d     r3.6<0;1,0>:d     1:w              
(W)     mul (1|M0)               acc0.0<1>:d   r3.6<0;1,0>:d     r3.4<0;1,0>:uw  
(W)     or (1|M0)                r2.15<1>:d    r3.6<0;1,0>:d     2:w              
(W)     mov (1|M0)               r2.4<1>:d     r8.0<0;1,0>:d                    {Compacted}
(W)     mov (1|M0)               r2.5<1>:d     r3.11<0;1,0>:d                  
(W)     macl (1|M0)              r5.0<1>:d     r3.6<0;1,0>:d     r3.2<0;1,0>:d    {Compacted}
(W)     mul (1|M0)               acc0.0<1>:d   r2.6<0;1,0>:d     r3.4<0;1,0>:uw   {I@6}
(W)     macl (1|M0)              r7.0<1>:d     r2.6<0;1,0>:d     r3.2<0;1,0>:d    {Compacted}
(W)     mul (1|M0)               acc0.0<1>:d   r2.15<0;1,0>:d    r3.4<0;1,0>:uw   {I@6}
(W)     shl (1|M0)               r3.6<1>:q     r2.2<0;1,0>:q     1:w               {I@5}
(W)     macl (1|M0)              r6.0<1>:d     r2.15<0;1,0>:d    r3.2<0;1,0>:d   
(W)     add (1|M0)               r2.5<1>:d     r2.7<0;1,0>:d     r5.0<0;1,0>:d    {I@6}
(W)     add (1|M0)               r2.4<1>:d     r2.7<0;1,0>:d     r7.0<0;1,0>:d    {I@5}
(W)     add (1|M0)               r1.12<1>:d    r2.7<0;1,0>:d     r6.0<0;1,0>:d    {I@3}
        add (32|M0)              r6.0<1>:d     r22.0<1;1,0>:d    r2.5<0;1,0>:d    {Compacted,I@3}
        mov (16|M0)              r19.0<2>:ud   r6.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M16)             r17.0<2>:ud   r7.0<1;1,0>:ud                   {Compacted}
        add3 (32|M0)             r6.0<1>:d     r22.0<1;0>:d      r2.5<0;0>:d       1:w              
        mov (16|M0)              r40.0<2>:ud   r6.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M16)             r38.0<2>:ud   r7.0<1;1,0>:ud                   {Compacted}
        add3 (32|M0)             r6.0<1>:d     r22.0<1;0>:d      r2.5<0;0>:d       2:w              
        mov (16|M0)              r44.0<2>:ud   r6.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M16)             r42.0<2>:ud   r7.0<1;1,0>:ud                   {Compacted}
        add3 (32|M0)             r6.0<1>:d     r22.0<1;0>:d      r2.5<0;0>:d       3:w              
        mov (16|M0)              r13.0<2>:ud   r6.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M16)             r36.0<2>:ud   r7.0<1;1,0>:ud                   {Compacted}
        add (32|M0)              r6.0<1>:d     r22.0<1;1,0>:d    r2.4<0;1,0>:d    {Compacted}
(W)     mov (1|M0)               r1.8<1>:ud    a0.2<0;1,0>:ud                  
        mov (16|M16)             r52.0<2>:ud   r7.0<1;1,0>:ud                   {Compacted,I@2}
(W)     shr (1|M0)               a0.2<1>:ud    r1.9<0;1,0>:ud    0x4:ud              {F@1}
        shl (16|M16)             r117.0<1>:q   r52.0<2;1,0>:d    1:w               {I@2}
        mov (16|M0)              r15.0<2>:ud   r6.0<1;1,0>:ud                   {Compacted}
(W)     send.ugm (1|M0)          null     r4  r117:2  0x83400000:a0.2        0x4200E504           {I@2,$6} // wr:1+2, rd:0; store.ugm.d32x32t.a32.ss[a0.2][A-0xF980]
        add3 (32|M0)             r6.0<1>:d     r22.0<1;0>:d      r2.4<0;0>:d       1:w              
        mov (16|M0)              r46.0<2>:ud   r6.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M16)             r74.0<2>:ud   r7.0<1;1,0>:ud                   {Compacted}
        add3 (32|M0)             r6.0<1>:d     r22.0<1;0>:d      r2.4<0;0>:d       2:w              
        mov (16|M0)              r72.0<2>:ud   r6.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M16)             r78.0<2>:ud   r7.0<1;1,0>:ud                   {Compacted}
        add3 (32|M0)             r6.0<1>:d     r22.0<1;0>:d      r2.4<0;0>:d       3:w              
        mov (16|M0)              r76.0<2>:ud   r6.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M16)             r82.0<2>:ud   r7.0<1;1,0>:ud                   {Compacted}
        add (32|M0)              r6.0<1>:d     r22.0<1;1,0>:d    r1.12<0;1,0>:d  
        mov (16|M0)              r80.0<2>:ud   r6.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M16)             r86.0<2>:ud   r7.0<1;1,0>:ud                   {Compacted}
        add3 (32|M0)             r6.0<1>:d     r22.0<1;0>:d      r1.12<0;0>:d      1:w              
(W)     or (1|M0)                r3.5<1>:d     r3.6<0;1,0>:d     3:w              
        mov (16|M0)              r84.0<2>:ud   r6.0<1;1,0>:ud                   {Compacted,I@2}
        mov (16|M16)             r90.0<2>:ud   r7.0<1;1,0>:ud                   {Compacted}
        add3 (32|M0)             r6.0<1>:d     r22.0<1;0>:d      r1.12<0;0>:d      2:w              
(W)     mul (1|M0)               acc0.0<1>:d   r3.5<0;1,0>:d     r3.4<0;1,0>:uw   {I@4}
(W)     macl (1|M0)              r5.0<1>:d     r3.5<0;1,0>:d     r3.2<0;1,0>:d   
        mov (16|M0)              r88.0<2>:ud   r6.0<1;1,0>:ud                   {Compacted,I@3}
        mov (16|M16)             r94.0<2>:ud   r7.0<1;1,0>:ud                   {Compacted}
        add3 (32|M0)             r6.0<1>:d     r22.0<1;0>:d      r1.12<0;0>:d      3:w              
(W)     add (1|M0)               r1.13<1>:d    r2.7<0;1,0>:d     r5.0<0;1,0>:d    {I@4}
        mov (16|M0)              r92.0<2>:ud   r6.0<1;1,0>:ud                   {Compacted,I@2}
        mov (16|M16)             r98.0<2>:ud   r7.0<1;1,0>:ud                   {Compacted}
        add (32|M0)              r6.0<1>:d     r22.0<1;1,0>:d    r1.13<0;1,0>:d   {I@3}
        shl (16|M16)             r117.0<1>:q   r74.0<2;1,0>:d    1:w               {$6.src}
        mov (16|M0)              r96.0<2>:ud   r6.0<1;1,0>:ud                   {Compacted,I@2}
        mov (16|M16)             r102.0<2>:ud  r7.0<1;1,0>:ud                   {Compacted}
        add3 (32|M0)             r6.0<1>:d     r22.0<1;0>:d      r1.13<0;0>:d      1:w              
(W)     send.ugm (1|M0)          null     r4  r117:2  0x83800000:a0.2        0x4200E504           {I@4,$7} // wr:1+2, rd:0; store.ugm.d32x32t.a32.ss[a0.2][A-0xF900]
        shl (16|M0)              r117.0<1>:q   r92.0<2;1,0>:d    1:w               {$7.src}
        mov (16|M0)              r100.0<2>:ud  r6.0<1;1,0>:ud                   {Compacted,I@2}
        mov (16|M16)             r106.0<2>:ud  r7.0<1;1,0>:ud                   {Compacted}
        add3 (32|M0)             r6.0<1>:d     r22.0<1;0>:d      r1.13<0;0>:d      2:w              
(W)     send.ugm (1|M0)          null     r4  r117:2  0x83C00000:a0.2        0x4200E504           {I@4,$8} // wr:1+2, rd:0; store.ugm.d32x32t.a32.ss[a0.2][A-0xF880]
        shl (16|M0)              r117.0<1>:q   r96.0<2;1,0>:d    1:w               {$8.src}
(W)     send.ugm (1|M0)          null     r4  r117:2  0x84000000:a0.2        0x4200E504           {I@1,$9} // wr:1+2, rd:0; store.ugm.d32x32t.a32.ss[a0.2][A-0xF800]
        add3 (32|M0)             r10.0<1>:d    r22.0<1;0>:d      r1.13<0;0>:d      3:w              
        mov (16|M0)              r104.0<2>:ud  r6.0<1;1,0>:ud                   {Compacted}
        shl (16|M0)              r117.0<1>:q   r100.0<2;1,0>:d   1:w               {$9.src}
(W)     send.ugm (1|M0)          null     r4  r117:2  0x84400000:a0.2        0x4200E504           {I@1,$10} // wr:1+2, rd:0; store.ugm.d32x32t.a32.ss[a0.2][A-0xF780]
        shl (16|M0)              r117.0<1>:q   r104.0<2;1,0>:d   1:w               {$10.src}
        mov (16|M0)              r5.0<2>:ud    r10.0<1;1,0>:ud                  {Compacted}
(W)     send.ugm (1|M0)          null     r4  r117:2  0x84800000:a0.2        0x4200E504           {I@2,$11} // wr:1+2, rd:0; store.ugm.d32x32t.a32.ss[a0.2][A-0xF700]
        sync.nop                             null                             {Compacted,$11.src}
(W)     send.ugm (1|M0)          r117     r4  null:0  0x83400000:a0.2        0x4220E500           {$12} // wr:1+0, rd:2; load.ugm.d32x32t.a32.ss[a0.2][A-0xF980]
(W)     send.ugm (1|M0)          null     r4  r5:2  0x82C00000:a0.2        0x4200E504           {I@1,$13} // wr:1+2, rd:0; store.ugm.d32x32t.a32.ss[a0.2][A-0xFA80]
        mov (16|M16)             r114.0<2>:ud  r7.0<1;1,0>:ud                   {Compacted}
        mov (16|M16)             r5.0<2>:ud    r11.0<1;1,0>:ud                  {Compacted,$13.src}
        shl (16|M16)             r7.0<1>:q     r42.0<2;1,0>:d    1:w              
(W)     send.ugm (1|M0)          null     r4  r5:4  0x82000000:a0.2        0x4200F504           {I@1,$14} // wr:1+4, rd:0; store.ugm.d32x64t.a32.ss[a0.2][A-0xFC00]
        shl (16|M16)             r5.0<1>:q     r36.0<2;1,0>:d    1:w               {$14.src}
(W)     send.ugm (1|M0)          null     r4  r5:2  0x83000000:a0.2        0x4200E504           {I@1,$15} // wr:1+2, rd:0; store.ugm.d32x32t.a32.ss[a0.2][A-0xFA00]
        shl (16|M0)              r5.0<1>:q     r15.0<2;1,0>:d    1:w               {$15.src}
(W)     add (1|M0)               r1.5<1>:q     r3.6<0;1,0>:q     r2.4<0;1,0>:q   
(W)     send.ugm (1|M0)          null     r4  r5:2  0x81C00000:a0.2        0x4200E504           {I@2,$0} // wr:1+2, rd:0; store.ugm.d32x32t.a32.ss[a0.2][A-0xFC80]
        shl (16|M0)              r5.0<1>:q     r46.0<2;1,0>:d    1:w               {$0.src}
        shl (16|M0)              r9.0<1>:q     r19.0<2;1,0>:d    1:w              
        shl (16|M16)             r19.0<1>:q    r90.0<2;1,0>:d    1:w              
(W)     send.ugm (1|M0)          null     r4  r5:2  0x82800000:a0.2        0x4200E504           {I@3,$1} // wr:1+2, rd:0; store.ugm.d32x32t.a32.ss[a0.2][A-0xFB00]
        shl (16|M16)             r11.0<1>:q    r17.0<2;1,0>:d    1:w              
(W)     send.ugm (1|M0)          null     r4  r9:4  0x80400000:a0.2        0x4200F504           {I@1,$2} // wr:1+4, rd:0; store.ugm.d32x64t.a32.ss[a0.2][A-0xFF80]
        shl (16|M16)             r36.0<1>:q    r86.0<2;1,0>:d    1:w              
        shl (16|M16)             r110.0<1>:q   r114.0<2;1,0>:d   1:w              
        shl (16|M0)              r9.0<1>:q     r40.0<2;1,0>:d    1:w               {$2.src}
        shl (16|M16)             r11.0<1>:q    r38.0<2;1,0>:d    1:w              
(W)     send.ugm (1|M0)          null     r4  r9:4  0x80C00000:a0.2        0x4200F504           {I@1,$3} // wr:1+4, rd:0; store.ugm.d32x64t.a32.ss[a0.2][A-0xFE80]
        shl (16|M0)              r74.0<1>:q    r72.0<2;1,0>:d    1:w              
        shl (16|M0)              r9.0<1>:q     r44.0<2;1,0>:d    1:w               {$3.src}
        shl (16|M0)              r11.0<1>:q    r13.0<2;1,0>:d    1:w              
(W)     send.ugm (1|M0)          null     r4  r9:4  0x81400000:a0.2        0x4200F504           {I@1,$4} // wr:1+4, rd:0; store.ugm.d32x64t.a32.ss[a0.2][A-0xFD80]
        shl (16|M0)              r46.0<1>:q    r80.0<2;1,0>:d    1:w              
        add (16|M0)              r80.0<1>:q    r1.5<0;1,0>:q     r74.0<1;1,0>:q   {Compacted}
        add (16|M16)             r74.0<1>:q    r1.5<0;1,0>:q     r36.0<1;1,0>:q   {Compacted}
        shl (16|M0)              r42.0<1>:q    r84.0<2;1,0>:d    1:w              
        add (16|M16)             r90.0<1>:q    r1.5<0;1,0>:q     r117.0<1;1,0>:q  {Compacted,$12.dst}
(W)     send.ugm (1|M0)          r117     r4  null:0  0x83800000:a0.2        0x4220E500           {I@1,$5} // wr:1+0, rd:2; load.ugm.d32x32t.a32.ss[a0.2][A-0xF900]
        sync.nop                             null                             {Compacted,$1.src}
(W)     send.ugm (1|M0)          r5       r4  null:0  0x82C00000:a0.2        0x4220E500           {$6} // wr:1+0, rd:2; load.ugm.d32x32t.a32.ss[a0.2][A-0xFA80]
        shl (16|M16)             r17.0<1>:q    r94.0<2;1,0>:d    1:w              
        shl (16|M16)             r44.0<1>:q    r78.0<2;1,0>:d    1:w              
        shl (16|M16)             r112.0<1>:q   r106.0<2;1,0>:d   1:w              
        shl (16|M16)             r40.0<1>:q    r82.0<2;1,0>:d    1:w              
        add (16|M16)             r82.0<1>:q    r1.5<0;1,0>:q     r44.0<1;1,0>:q   {Compacted,I@3}
        add (16|M0)              r44.0<1>:q    r1.5<0;1,0>:q     r42.0<1;1,0>:q   {Compacted}
        add (16|M16)             r42.0<1>:q    r1.5<0;1,0>:q     r17.0<1;1,0>:q   {Compacted}
        shl (16|M0)              r72.0<1>:q    r76.0<2;1,0>:d    1:w              
        shl (16|M16)             r13.0<1>:q    r102.0<2;1,0>:d   1:w              
        add (16|M0)              r76.0<1>:q    r1.5<0;1,0>:q     r72.0<1;1,0>:q   {Compacted,I@2}
        add (16|M0)              r72.0<1>:q    r1.5<0;1,0>:q     r46.0<1;1,0>:q   {Compacted}
        sync.nop                             null                             {Compacted,$4.src}
(W)     send.ugm (1|M0)          r9       r4  null:0  0x80400000:a0.2        0x4240F500           {$7} // wr:1+0, rd:4; load.ugm.d32x64t.a32.ss[a0.2][A-0xFF80]
        add (16|M16)             r46.0<1>:q    r1.5<0;1,0>:q     r19.0<1;1,0>:q   {Compacted}
        add (16|M16)             r19.0<1>:q    r1.5<0;1,0>:q     r13.0<1;1,0>:q   {Compacted,I@4}
        shl (16|M0)              r38.0<1>:q    r88.0<2;1,0>:d    1:w              
        shl (16|M16)             r15.0<1>:q    r98.0<2;1,0>:d    1:w              
        add (16|M16)             r78.0<1>:q    r1.5<0;1,0>:q     r40.0<1;1,0>:q   {Compacted}
        add (16|M0)              r40.0<1>:q    r1.5<0;1,0>:q     r38.0<1;1,0>:q   {Compacted,I@3}
        add (16|M16)             r38.0<1>:q    r1.5<0;1,0>:q     r15.0<1;1,0>:q   {Compacted,I@3}
        add (16|M16)             r15.0<1>:q    r1.5<0;1,0>:q     r112.0<1;1,0>:q  {Compacted}
        send.ugm (32|M0)         r50      r80  null:0  0x0            0x08200B80           {F@7,$8} // wr:4+0, rd:2; load.ugm.d16u32.a64
        send.ugm (32|M0)         r32      r72  null:0  0x0            0x08200B80           {F@5,$9} // wr:4+0, rd:2; load.ugm.d16u32.a64
        send.ugm (32|M0)         r30      r44  null:0  0x0            0x08200B80           {F@4,$10} // wr:4+0, rd:2; load.ugm.d16u32.a64
        add (16|M16)             r86.0<1>:q    r1.5<0;1,0>:q     r117.0<1;1,0>:q  {Compacted,$5.dst}
(W)     send.ugm (1|M0)          r117     r4  null:0  0x83C00000:a0.2        0x4220E500           {I@1,$11} // wr:1+0, rd:2; load.ugm.d32x32t.a32.ss[a0.2][A-0xF880]
        shl (16|M0)              r114.0<1>:q   r5.0<2;1,0>:d     1:w               {$6.dst}
(W)     send.ugm (1|M0)          r5       r4  null:0  0x82000000:a0.2        0x4240F500           {I@1,$13} // wr:1+0, rd:4; load.ugm.d32x64t.a32.ss[a0.2][A-0xFC00]
        send.ugm (32|M0)         r34      r76  null:0  0x0            0x08200B80           {$14} // wr:4+0, rd:2; load.ugm.d16u32.a64
        send.ugm (32|M0)         r28      r40  null:0  0x0            0x08200B80           {F@3,$15} // wr:4+0, rd:2; load.ugm.d16u32.a64
        add (16|M0)              r104.0<1>:q   r1.5<0;1,0>:q     r9.0<1;1,0>:q    {Compacted,$7.dst}
        add (16|M16)             r106.0<1>:q   r1.5<0;1,0>:q     r11.0<1;1,0>:q   {Compacted}
(W)     send.ugm (1|M0)          r9       r4  null:0  0x80C00000:a0.2        0x4240F500           {I@1,$0} // wr:1+0, rd:4; load.ugm.d32x64t.a32.ss[a0.2][A-0xFE80]
        send.ugm (32|M0)         r62      r104  null:0  0x0            0x08200B80           {$12} // wr:4+0, rd:2; load.ugm.d16u32.a64
        add (16|M0)              r36.0<1>:q    r1.5<0;1,0>:q     r117.0<1;1,0>:q  {Compacted,$11.dst}
(W)     send.ugm (1|M0)          r117     r4  null:0  0x84000000:a0.2        0x4220E500           {I@1,$2} // wr:1+0, rd:2; load.ugm.d32x32t.a32.ss[a0.2][A-0xF800]
        shl (16|M16)             r52.0<1>:q    r5.0<2;1,0>:d     1:w               {$13.dst}
(W)     send.ugm (1|M0)          r5       r4  null:0  0x83000000:a0.2        0x4220E500           {I@1,$1} // wr:1+0, rd:2; load.ugm.d32x32t.a32.ss[a0.2][A-0xFA00]
        add (16|M16)             r98.0<1>:q    r1.5<0;1,0>:q     r7.0<1;1,0>:q    {Compacted}
        add (16|M16)             r7.0<1>:q     r1.5<0;1,0>:q     r52.0<1;1,0>:q   {Compacted}
        send.ugm (32|M0)         r26      r36  null:0  0x0            0x08200B80           {F@2,$3} // wr:4+0, rd:2; load.ugm.d16u32.a64
        add (16|M0)              r100.0<1>:q   r1.5<0;1,0>:q     r9.0<1;1,0>:q    {Compacted,$0.dst}
        add (16|M16)             r102.0<1>:q   r1.5<0;1,0>:q     r11.0<1;1,0>:q   {Compacted}
(W)     send.ugm (1|M0)          r9       r4  null:0  0x81400000:a0.2        0x4240F500           {I@1,$4} // wr:1+0, rd:4; load.ugm.d32x64t.a32.ss[a0.2][A-0xFD80]
        send.ugm (32|M0)         r60      r100  null:0  0x0            0x08200B80           {$5} // wr:4+0, rd:2; load.ugm.d16u32.a64
        shl (16|M0)              r70.0<1>:ud   r62.0<2;1,0>:uw   0x10:uw              {$12.dst}
        shl (16|M16)             r71.0<1>:ud   r63.0<2;1,0>:uw   0x10:uw             
        add (16|M0)              r17.0<1>:q    r1.5<0;1,0>:q     r117.0<1;1,0>:q  {Compacted,$2.dst}
(W)     send.ugm (1|M0)          r117     r4  null:0  0x84400000:a0.2        0x4220E500           {I@1,$6} // wr:1+0, rd:2; load.ugm.d32x32t.a32.ss[a0.2][A-0xF780]
        add (16|M16)             r94.0<1>:q    r1.5<0;1,0>:q     r5.0<1;1,0>:q    {Compacted,$1.dst}
(W)     send.ugm (1|M0)          r5       r4  null:0  0x81C00000:a0.2        0x4220E500           {I@1,$7} // wr:1+0, rd:2; load.ugm.d32x32t.a32.ss[a0.2][A-0xFC80]
        send.ugm (32|M0)         r24      r17  null:0  0x0            0x08200B80           {F@1,$11} // wr:4+0, rd:2; load.ugm.d16u32.a64
        add (16|M0)              r96.0<1>:q    r1.5<0;1,0>:q     r9.0<1;1,0>:q    {Compacted,$4.dst}
        add (16|M0)              r92.0<1>:q    r1.5<0;1,0>:q     r11.0<1;1,0>:q   {Compacted}
        add (16|M16)             r11.0<1>:q    r1.5<0;1,0>:q     r110.0<1;1,0>:q  {Compacted}
        send.ugm (32|M0)         r58      r96  null:0  0x0            0x08200B80           {I@3,$12} // wr:4+0, rd:2; load.ugm.d16u32.a64
        send.ugm (32|M0)         r56      r92  null:0  0x0            0x08200B80           {I@2,$13} // wr:4+0, rd:2; load.ugm.d16u32.a64
        shl (16|M0)              r68.0<1>:ud   r60.0<2;1,0>:uw   0x10:uw              {$5.dst}
        shl (16|M16)             r69.0<1>:ud   r61.0<2;1,0>:uw   0x10:uw             
        add (16|M0)              r13.0<1>:q    r1.5<0;1,0>:q     r117.0<1;1,0>:q  {Compacted,$6.dst}
(W)     send.ugm (1|M0)          r117     r4  null:0  0x84800000:a0.2        0x4220E500           {I@1,$0} // wr:1+0, rd:2; load.ugm.d32x32t.a32.ss[a0.2][A-0xF700]
        add (16|M0)              r88.0<1>:q    r1.5<0;1,0>:q     r5.0<1;1,0>:q    {Compacted,$7.dst}
(W)     send.ugm (1|M0)          r5       r4  null:0  0x82800000:a0.2        0x4220E500           {I@1,$1} // wr:1+0, rd:2; load.ugm.d32x32t.a32.ss[a0.2][A-0xFB00]
        send.ugm (32|M0)         r18      r13  null:0  0x0            0x08200B80           {$2} // wr:4+0, rd:2; load.ugm.d16u32.a64
        send.ugm (32|M0)         r54      r88  null:0  0x0            0x08200B80           {$4} // wr:4+0, rd:2; load.ugm.d16u32.a64
        sync.nop                             null                             {Compacted,$1.src}
(W)     mov (1|M0)               a0.2<1>:ud    r1.8<0;1,0>:ud                   {$0.src}
        shl (16|M0)              r66.0<1>:ud   r58.0<2;1,0>:uw   0x10:uw              {$12.dst}
        shl (16|M16)             r67.0<1>:ud   r59.0<2;1,0>:uw   0x10:uw             
        shl (16|M0)              r58.0<1>:ud   r50.0<2;1,0>:uw   0x10:uw              {$8.dst}
        shl (16|M0)              r64.0<1>:ud   r56.0<2;1,0>:uw   0x10:uw              {$13.dst}
        shl (16|M16)             r65.0<1>:ud   r57.0<2;1,0>:uw   0x10:uw             
        shl (16|M16)             r59.0<1>:ud   r51.0<2;1,0>:uw   0x10:uw             
        shl (16|M0)              r50.0<1>:ud   r30.0<2;1,0>:uw   0x10:uw              {$10.dst}
        shl (16|M0)              r56.0<1>:ud   r34.0<2;1,0>:uw   0x10:uw              {$14.dst}
        shl (16|M16)             r57.0<1>:ud   r35.0<2;1,0>:uw   0x10:uw             
        shl (16|M16)             r51.0<1>:ud   r31.0<2;1,0>:uw   0x10:uw             
        shl (16|M0)              r30.0<1>:ud   r24.0<2;1,0>:uw   0x10:uw              {$11.dst}
        shl (16|M0)              r34.0<1>:ud   r28.0<2;1,0>:uw   0x10:uw              {$15.dst}
        add (16|M0)              r9.0<1>:q     r1.5<0;1,0>:q     r117.0<1;1,0>:q  {Compacted,$0.dst}
        add (16|M0)              r84.0<1>:q    r1.5<0;1,0>:q     r5.0<1;1,0>:q    {Compacted,$1.dst}
        add (16|M0)              r5.0<1>:q     r1.5<0;1,0>:q     r114.0<1;1,0>:q  {Compacted}
        send.ugm (32|M0)         r14      r9  null:0  0x0            0x08200B80           {I@3,$5} // wr:4+0, rd:2; load.ugm.d16u32.a64
        send.ugm (32|M0)         r52      r84  null:0  0x0            0x08200B80           {I@2,$6} // wr:4+0, rd:2; load.ugm.d16u32.a64
        send.ugm (32|M0)         r10      r5  null:0  0x0            0x08200B80           {I@1,$7} // wr:4+0, rd:2; load.ugm.d16u32.a64
        shl (16|M16)             r35.0<1>:ud   r29.0<2;1,0>:uw   0x10:uw             
        shl (16|M16)             r31.0<1>:ud   r25.0<2;1,0>:uw   0x10:uw             
        shl (16|M0)              r28.0<1>:ud   r18.0<2;1,0>:uw   0x10:uw              {$2.dst}
        shl (16|M0)              r62.0<1>:ud   r54.0<2;1,0>:uw   0x10:uw              {$4.dst}
        shl (16|M16)             r63.0<1>:ud   r55.0<2;1,0>:uw   0x10:uw             
        shl (16|M16)             r29.0<1>:ud   r19.0<2;1,0>:uw   0x10:uw             
        shl (16|M0)              r54.0<1>:ud   r32.0<2;1,0>:uw   0x10:uw              {$9.dst}
        shl (16|M16)             r55.0<1>:ud   r33.0<2;1,0>:uw   0x10:uw             
        shl (16|M0)              r32.0<1>:ud   r26.0<2;1,0>:uw   0x10:uw              {$3.dst}
        shl (16|M16)             r33.0<1>:ud   r27.0<2;1,0>:uw   0x10:uw             
        shl (16|M0)              r26.0<1>:ud   r14.0<2;1,0>:uw   0x10:uw              {$5.dst}
        shl (16|M16)             r27.0<1>:ud   r15.0<2;1,0>:uw   0x10:uw             
        shl (16|M0)              r60.0<1>:ud   r52.0<2;1,0>:uw   0x10:uw              {$6.dst}
        shl (16|M16)             r61.0<1>:ud   r53.0<2;1,0>:uw   0x10:uw             
        shl (16|M0)              r24.0<1>:ud   r10.0<2;1,0>:uw   0x10:uw              {$7.dst}
        shl (16|M16)             r25.0<1>:ud   r11.0<2;1,0>:uw   0x10:uw             
L4736:
(W)     add (1|M0)               r6.0<1>:q     r1.7<0;1,0>:q     r2.5<0;1,0>:q    {Compacted}
(W)     send.ugm (1|M0)          r86      r6  null:0  0x0            0x02109580           {I@1,$8} // wr:1+0, rd:1; load.ugm.d32x2t.a64
(W)     cmp (32|M0)   (lt)f0.0   null<1>:d     r86.0<0;1,0>:d    r86.1<0;1,0>:d   {$8.dst}
(W&~f0.0) jmpi                               L14048                                
L4792:
(W)     mov (1|M0)               r1.10<1>:f    r3.2<0;1,0>:d                   
        cmp (32|M0)   (eq)f3.0   null<2>:w     r48.0<1;1,0>:w    0:w              
(W)     math.sqt (1|M0)          r4.7<1>:f     r1.10<0;1,0>:f                   {F@1}
(W)     mov (1|M0)               r1.5<1>:q     r3.0<0;1,0>:d                    {M@1}
(W)     mov (1|M0)               r4.5<1>:ud    0x3F317200:ud                             
(W)     mov (1|M0)               r4.4<1>:ud    0x35BFBE8E:ud                             
(W)     mov (1|M0)               r4.3<1>:ud    0xBF000000:ud                             
(W)     mov (1|M0)               r4.2<1>:ud    0x3EAAAA83:ud                             
(W)     mov (1|M0)               r3.15<1>:ud   0xBE7FFF78:ud                             
(W)     mov (1|M0)               r3.14<1>:f    0x3E4CE814:f                              
(W)     mov (1|M0)               r3.13<1>:f    0xBE2ACEE6:f                              
(W)     mov (1|M0)               r3.11<1>:f    1.400587e-01:f                              
(W)     mov (1|M0)               r3.5<1>:f     0xBDF9889E:f                              
(W)     mov (1|M0)               r2.15<1>:f    0x3E0F335D:f                              
(W)     mov (1|M0)               r2.11<1>:f    1.0:f                              
(W)     mov (1|M0)               r1.15<1>:d    r86.0<0;1,0>:d                  
(W)     mov (1|M0)               r3.0<1>:f     0xBE0402C8:f                               {I@7}
(W)     mov (2|M0)               r1.12<1>:d    r1.10<1;1,0>:d                  
L5080:
(W)     mul (1|M0)               acc0.0<1>:d   r1.15<0;1,0>:d    r3.6<0;1,0>:uw   {I@2}
(W)     macl (1|M0)              r5.0<1>:d     r1.15<0;1,0>:d    r3.3<0;1,0>:d    {$1.src}
(W)     add (1|M0)               r2.5<1>:d     r5.0<0;1,0>:d     r3.7<0;1,0>:d    {I@1}
(W)     shl (1|M0)               r4.4<1>:q     r2.5<0;1,0>:d     1:w               {I@1}
(W)     add (1|M0)               r8.0<1>:q     r4.4<0;1,0>:q     r2.0<0;1,0>:q    {Compacted,I@1}
(W)     add (1|M0)               r6.0<1>:q     r4.4<0;1,0>:q     r2.1<0;1,0>:q    {Compacted}
(W)     send.ugm (1|M0)          r7       r8  null:0  0x0            0x04100B80           {I@2,$3} // wr:2+0, rd:1; load.ugm.d16u32.a64
(W)     send.ugm (1|M0)          r5       r6  null:0  0x0            0x04100B80           {I@1,$3} // wr:2+0, rd:1; load.ugm.d16u32.a64
(W)     shl (1|M0)               r4.1<1>:ud    r7.0<0;1,0>:uw    0x10:uw             
(W)     shl (1|M0)               r4.6<1>:ud    r5.0<0;1,0>:uw    0x10:uw              {$3.dst}
(W)     mul (1|M0)               r4.8<1>:f     r4.1<0;1,0>:f     -1.442695e+00:f               {I@2}
(W)     add (1|M0)               r4.6<1>:f     r4.6<0;1,0>:f     r4.14<0;1,0>:f   {I@1}
(W)     rndz (1|M0)              r5.0<1>:f     r4.8<0;1,0>:f                    {Compacted,F@2}
(W)     mov (1|M0)               r4.16<1>:bf   r4.6<0;1,0>:f                    {F@2}
(W)     mad (1|M0)               r4.6<1>:f     -r4.1<0;0>:f      r3.10<0;0>:f      r5.0<0>:f        {F@2}
(W)     math.exp (1|M0)          r6.0<1>:f     r5.0<0;1,0>:f                   
(W)     mad (1|M0)               r4.6<1>:f     r4.6<0;0>:f       r3.9<0;0>:f       r5.0<0>:f        {F@1}
(W)     cmp (1|M0)    (gt)f2.0   null<1>:f     r4.1<0;1,0>:f     105.0:f              
(W)     mul (1|M0)               r5.1<1>:f     r4.6<0;1,0>:f     1.442695e+00:f               {F@2}
(W)     cmp (1|M0)    (lt)f1.0   null<1>:f     r4.1<0;1,0>:f     -105.0:f              
(W)     math.exp (1|M0)          r6.1<1>:f     r5.1<0;1,0>:f                    {F@2}
(W)     mad (1|M0)               r4.6<1>:f     r2.11<0;0>:f      r6.0<0;0>:f       r6.1<0>:f        {M@1}
(W)     shl (1|M0)               r4.1<1>:ud    r4.16<0;1,0>:uw   0x10:uw              {F@2}
(W)     math.inv (1|M0)          r4.6<1>:f     r4.6<0;1,0>:f                    {F@1}
(W)     cmp (32|M0)   (lt)f0.0   null<1>:f     r4.1<0;1,0>:f     20.0:f               {I@1}
(W&~f2.0) sel (1|M0)             r4.6<1>:f     r4.6<0;1,0>:f     1.0:f               {M@1}
(W&~f1.0) sel (1|M0)             r2.6<1>:f     r4.6<0;1,0>:f     0.0:f               {F@1}
(W&~f0.0) jmpi                               L6168                                
L5504:
(W)     mul (1|M0)               r4.6<1>:f     r4.1<0;1,0>:f     1.442695e+00:f              
(W)     cmp (1|M0)    (lt)f0.0   null<1>:f     r4.1<0;1,0>:f     -105.0:f               {I@1}
(W)     rndz (1|M0)              r5.0<1>:f     r4.6<0;1,0>:f                    {Compacted,F@2}
(W)     cmp (1|M0)    (gt)f2.0   null<1>:f     r4.1<0;1,0>:f     105.0:f              
(W)     mad (1|M0)               r4.6<1>:f     r4.1<0;0>:f       r3.10<0;0>:f      r5.0<0>:f        {F@2}
(W)     math.exp (1|M0)          r6.0<1>:f     r5.0<0;1,0>:f                   
(W)     mad (1|M0)               r4.6<1>:f     r4.6<0;0>:f       r3.9<0;0>:f       r5.0<0>:f        {F@1}
(W)     mul (1|M0)               r5.1<1>:f     r4.6<0;1,0>:f     1.442695e+00:f               {F@1}
(W)     math.exp (1|M0)          r6.1<1>:f     r5.1<0;1,0>:f                    {F@1}
(W)     mad (1|M0)               r4.6<1>:f     r2.11<0;0>:f      r6.0<0;0>:f       r6.1<0>:f        {M@1}
(W&~f0.0) sel (1|M0)             r4.6<1>:f     r4.6<0;1,0>:f     1.0:f               {F@1}
(W&~f2.0) sel (1|M0)             r4.6<1>:f     r4.6<0;1,0>:f     inf:f               {F@1}
(W)     cmp (32|M0)   (gt)f0.0   null<1>:f     r4.6<0;1,0>:f     0.0:f               {F@1}
(W)     and (1|M0)               r4.8<1>:d     r4.6<0;1,0>:d     2147483647:d              
(W&f0.0) cmp (32|M0)  (lt)f0.0   null<1>:f     r4.8<0;1,0>:f     inf:f               {I@1}
(W&f0.0) jmpi                                L5784                                
L5752:
(W)     math.log (1|M0)          r4.1<1>:f     r4.6<0;1,0>:f                   
(W)     jmpi                                 L6168                                
L5784:
(W)     cmp (32|M0)   (lt)f1.0   null<1>:f     r4.6<0;1,0>:f     0x800000:f              
(W)     mul (1|M0)               r4.9<1>:f     r4.6<0;1,0>:f     8.388608e+06:f              
(W)     mov (1|M0)               r4.8<1>:ud    0xC1B80000:ud                              {F@3}
(W&f1.0) sel (1|M0)              r4.6<1>:f     r4.9<0;1,0>:f     r4.6<0;1,0>:f    {A@1}
(W&f1.0) sel (1|M0)              r4.11<1>:f    r4.8<0;1,0>:f     0.0:f               {I@1}
(W)     add (1|M0)               r4.6<1>:d     r4.6<0;1,0>:d     -1059760811:d               {F@2}
(W)     and (1|M0)               r4.8<1>:d     r4.6<0;1,0>:d     8388607:d               {A@1}
(W)     asr (1|M0)               r4.6<1>:d     r4.6<0;1,0>:d     23:w              
(W)     add (1|M0)               r4.9<1>:d     r4.8<0;1,0>:d     1059760811:d               {I@2}
(W)     mov (1|M0)               r4.6<1>:f     r4.6<0;1,0>:d                    {I@2}
(W)     add (1|M0)               r4.8<1>:f     r4.11<0;1,0>:f    r4.6<0;1,0>:f    {A@1}
(W)     add (1|M0)               r4.6<1>:f     r4.9<0;1,0>:f     -1.0:f              
(W)     mad (1|M0)               r4.9<1>:f     r2.15<0;0>:f      r3.0<0;0>:f       r4.6<0>:f        {F@1}
(W)     mad (1|M0)               r4.9<1>:f     r3.5<0;0>:f       r4.6<0;0>:f       r4.9<0>:f        {F@1}
(W)     mad (1|M0)               r4.9<1>:f     r3.11<0;0>:f      r4.6<0;0>:f       r4.9<0>:f        {F@1}
(W)     mad (1|M0)               r4.9<1>:f     r3.13<0;0>:f      r4.6<0;0>:f       r4.9<0>:f        {F@1}
(W)     mad (1|M0)               r4.9<1>:f     r3.14<0;0>:f      r4.6<0;0>:f       r4.9<0>:f        {F@1}
(W)     mad (1|M0)               r4.9<1>:f     r3.15<0;0>:f      r4.6<0;0>:f       r4.9<0>:f        {F@1}
(W)     mad (1|M0)               r4.9<1>:f     r4.2<0;0>:f       r4.6<0;0>:f       r4.9<0>:f        {F@1}
(W)     mad (1|M0)               r4.9<1>:f     r4.3<0;0>:f       r4.6<0;0>:f       r4.9<0>:f        {F@1}
(W)     mul (1|M0)               r4.9<1>:f     r4.6<0;1,0>:f     r4.9<0;1,0>:f    {F@1}
(W)     mad (1|M0)               r4.6<1>:f     r4.6<0;0>:f       r4.6<0;0>:f       r4.9<0>:f        {F@1}
(W)     mad (1|M0)               r4.6<1>:f     r4.6<0;0>:f       r4.4<0;0>:f       r4.8<0>:f        {F@1}
(W)     mad (1|M0)               r4.1<1>:f     r4.6<0;0>:f       r4.5<0;0>:f       r4.8<0>:f        {F@1}
L6168:
(W)     mul (1|M0)               r4.1<1>:f     r4.1<0;1,0>:f     -r4.10<0;1,0>:f  {F@1}
(W)     cmp (32|M0)   (eq)f1.0   null<1>:d     r3.8<0;1,0>:d     0:w              
(W)     mul (1|M0)               r4.6<1>:f     r4.1<0;1,0>:f     1.442695e+00:f               {F@1}
(W)     mul (1|M0)               acc0.0<1>:d   r1.15<0;1,0>:d    r3.2<0;1,0>:uw  
(W)     rndz (1|M0)              r2.4<1>:f     r4.6<0;1,0>:f                    {F@1}
(W)     macl (1|M0)              r6.0<1>:d     r1.15<0;1,0>:d    r3.1<0;1,0>:d   
(W)     mad (1|M0)               r4.6<1>:f     r4.1<0;0>:f       r3.10<0;0>:f      r2.4<0>:f        {F@1}
(W)     mad (1|M0)               r4.6<1>:f     r4.6<0;0>:f       r3.9<0;0>:f       r2.4<0>:f        {F@1}
(W)     mul (1|M0)               r2.10<1>:f    r4.6<0;1,0>:f     1.442695e+00:f               {F@1}
(W&~f1.0) jmpi                               L6360                                
L6328:
(W)     mov (1|M0)               r86.6<1>:d    -1:w                              
(W)     jmpi                                 L6840                                
L6360:
(W)     asr (1|M0)               r4.11<1>:d    r3.8<0;1,0>:d     31:w              
(W)     add (1|M0)               r4.6<1>:d     r4.11<0;1,0>:d    r3.8<0;1,0>:d    {A@1}
(W)     xor (1|M0)               r4.6<1>:d     r4.6<0;1,0>:d     r4.11<0;1,0>:d   {I@1}
(W)     xor (1|M0)               cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x30:uw              {A@1}
(W)     mov (1|M0)               r86.2<1>:f    r4.6<0;1,0>:ud                   {A@1}
(W)     mov (1|M0)               r86.4<1>:f    0xB4C00000:f                               {Compacted}
(W)     math.inv (1|M0)          r86.3<1>:f    r86.2<0;1,0>:f                   {F@2}
(W)     mov (1|M0)               r4.15<1>:f    r3.7<0;1,0>:ud                  
(W)     mad (1|M0)               r86.5<1>:f    r86.3<0;0>:f      r86.4<0;0>:f      r86.3<0>:f       {A@1}
(W)     mov (1|M0)               r4.8<1>:ud    r86.2<0;1,0>:f                  
(W)     mov (1|M0)               r86.3<1>:ud   r4.15<0;1,0>:f                   {F@1}
(W)     mul (1|M0)               r86.4<1>:f    r4.15<0;1,0>:f    r86.5<0;1,0>:f  
(W)     add (1|M0)               r4.12<1>:d    r4.6<0;1,0>:d     -r4.8<0;1,0>:d   {I@2}
(W)     add (1|M0)               r4.13<1>:d    r3.7<0;1,0>:d     -r86.3<0;1,0>:d  {I@2}
(W)     mov (1|M0)               r86.4<1>:ud   r86.4<0;1,0>:f                   {F@1}
(W)     mov (1|M0)               r4.8<1>:f     r4.12<0;1,0>:ud                  {I@3}
(W)     mov (1|M0)               r4.9<1>:f     r4.13<0;1,0>:ud                  {I@2}
(W)     mov (1|M0)               r86.3<1>:f    r86.4<0;1,0>:ud                  {I@1}
(W)     mad (1|M0)               r4.12<1>:f    r4.15<0;0>:f      r86.3<0;0>:f      -r86.2<0>:f      {F@1}
(W)     mad (1|M0)               r4.8<1>:f     r4.9<0;0>:f       r86.3<0;0>:f      -r4.8<0>:f      
(W)     add (1|M0)               r4.8<1>:f     r4.12<0;1,0>:f    r4.8<0;1,0>:f    {F@1}
(W)     mul (1|M0)               r4.8<1>:f     r86.5<0;1,0>:f    r4.8<0;1,0>:f    {F@1}
(W)     xor (1|M0)               cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x30:uw              {A@1}
(W)     mov (1|M0)               r4.8<1>:ud    r4.8<0;1,0>:f                    {A@1}
(W)     add (1|M0)               r4.8<1>:d     r4.8<0;1,0>:d     r86.4<0;1,0>:d   {I@1}
(W)     mul (1|M0)               acc0.0<1>:d   r4.8<0;1,0>:d     r4.12<0;1,0>:uw  {I@1}
(W)     macl (1|M0)              r5.0<1>:d     r4.8<0;1,0>:d     r4.6<0;1,0>:d    {Compacted}
(W)     add (1|M0)               r4.9<1>:d     r3.7<0;1,0>:d     -r5.0<0;1,0>:d   {I@1}
(W)     cmp (1|M0)    (ge)f0.0   r4.6<1>:ud    r4.9<0;1,0>:ud    r4.6<0;1,0>:ud   {I@1}
(W)     add3 (1|M0)              r4.6<1>:d     r4.8<0;0>:d       r4.11<0;0>:d      -r4.6<0>:d       {I@1}
(W)     xor (1|M0)               r86.6<1>:d    r4.6<0;1,0>:d     r4.11<0;1,0>:d   {I@1}
L6840:
(W)     add (1|M0)               r4.6<1>:d     r6.0<0;1,0>:d     r86.6<0;1,0>:d   {I@1}
        or (32|M0)               r52.0<1>:d    r22.0<1;1,0>:d    1:w               {Compacted}
(W)     mul (1|M0)               acc0.0<1>:d   r4.6<0;1,0>:d     r3.4<0;1,0>:uw   {I@2}
(W)     macl (1|M0)              r5.0<1>:d     r4.6<0;1,0>:d     r3.2<0;1,0>:d    {Compacted}
        or (32|M0)               r72.0<1>:d    r22.0<1;1,0>:d    2:w               {Compacted}
        add (32|M0)              r6.0<1>:d     r5.0<0;1,0>:d     r52.0<1;1,0>:d   {Compacted,I@2}
        or (32|M0)               r74.0<1>:d    r22.0<1;1,0>:d    3:w               {Compacted}
        add (32|M0)              r8.0<1>:d     r5.0<0;1,0>:d     r22.0<1;1,0>:d   {Compacted}
        add (32|M0)              r10.0<1>:d    r5.0<0;1,0>:d     r72.0<1;1,0>:d   {Compacted,I@4}
        mov (16|M0)              r12.0<2>:ud   r6.0<1;1,0>:ud                   {Compacted,I@4}
        mov (16|M16)             r14.0<2>:ud   r7.0<1;1,0>:ud                   {Compacted}
        add (32|M0)              r18.0<1>:d    r5.0<0;1,0>:d     r74.0<1;1,0>:d   {Compacted,I@5}
        mov (16|M16)             r36.0<2>:ud   r9.0<1;1,0>:ud                   {Compacted,I@5}
        mov (16|M0)              r5.0<2>:ud    r8.0<1;1,0>:ud                   {Compacted}
        shl (16|M0)              r84.0<1>:q    r12.0<2;1,0>:d    1:w               {I@5}
        shl (16|M16)             r82.0<1>:q    r14.0<2;1,0>:d    1:w               {I@5}
        mov (16|M0)              r7.0<2>:ud    r10.0<1;1,0>:ud                  {Compacted}
        shl (16|M0)              r15.0<1>:q    r5.0<2;1,0>:d     1:w               {I@4}
        shl (16|M16)             r13.0<1>:q    r36.0<2;1,0>:d    1:w              
        shl (16|M0)              r80.0<1>:q    r7.0<2;1,0>:d     1:w               {@3,$9.src}
        add (16|M0)              r5.0<1>:q     r84.0<1;1,0>:q    r1.2<0;1,0>:q    {Compacted}
        mov (16|M16)             r36.0<2>:ud   r11.0<1;1,0>:ud                  {Compacted}
        add (16|M16)             r7.0<1>:q     r82.0<1;1,0>:q    r1.2<0;1,0>:q    {Compacted}
        mov (16|M16)             r38.0<2>:ud   r19.0<1;1,0>:ud                  {Compacted}
        add (16|M0)              r9.0<1>:q     r15.0<1;1,0>:q    r1.2<0;1,0>:q    {Compacted,I@7}
        add (16|M16)             r11.0<1>:q    r13.0<1;1,0>:q    r1.2<0;1,0>:q    {Compacted,I@7}
        shl (16|M16)             r78.0<1>:q    r36.0<2;1,0>:d    1:w               {@5,$14.src}
        send.ugm (32|M0)         r36      r5  null:0  0x0            0x08200B80           {I@1,$4} // wr:4+0, rd:2; load.ugm.d16u32.a64
        shl (16|M16)             r44.0<1>:q    r38.0<2;1,0>:d    1:w               {$11.src}
        send.ugm (32|M0)         r38      r9  null:0  0x0            0x08200B80           {I@1,$5} // wr:4+0, rd:2; load.ugm.d16u32.a64
        mov (16|M0)              r40.0<2>:ud   r18.0<1;1,0>:ud                  {Compacted,$13.src}
        add (16|M0)              r5.0<1>:q     r80.0<1;1,0>:q    r1.2<0;1,0>:q    {Compacted,$4.src}
        add (16|M16)             r7.0<1>:q     r78.0<1;1,0>:q    r1.2<0;1,0>:q    {Compacted}
        shl (16|M0)              r76.0<1>:q    r40.0<2;1,0>:d    1:w               {@3,$15.src}
        send.ugm (32|M0)         r40      r5  null:0  0x0            0x08200B80           {I@1,$6} // wr:4+0, rd:2; load.ugm.d16u32.a64
        add (16|M16)             r11.0<1>:q    r44.0<1;1,0>:q    r1.2<0;1,0>:q    {Compacted,$5.src}
        add (16|M0)              r9.0<1>:q     r76.0<1;1,0>:q    r1.2<0;1,0>:q    {Compacted}
        add (16|M0)              r5.0<1>:q     r15.0<1;1,0>:q    r1.1<0;1,0>:q    {Compacted,$6.src}
        add (16|M16)             r7.0<1>:q     r13.0<1;1,0>:q    r1.1<0;1,0>:q    {Compacted}
        send.ugm (32|M0)         r42      r9  null:0  0x0            0x08200B80           {I@3,$7} // wr:4+0, rd:2; load.ugm.d16u32.a64
        add (16|M16)             r19.0<1>:q    r82.0<1;1,0>:q    r1.1<0;1,0>:q    {Compacted}
        add (16|M0)              r17.0<1>:q    r84.0<1;1,0>:q    r1.1<0;1,0>:q    {Compacted}
        add (16|M16)             r11.0<1>:q    r44.0<1;1,0>:q    r1.1<0;1,0>:q    {Compacted,$7.src}
        send.ugm (32|M0)         r44      r5  null:0  0x0            0x08200B80           {I@1,$8} // wr:4+0, rd:2; load.ugm.d16u32.a64
        send.ugm (32|M0)         r6       r17  null:0  0x0            0x08200B80           {$3} // wr:4+0, rd:2; load.ugm.d16u32.a64
        add (16|M16)             r15.0<1>:q    r78.0<1;1,0>:q    r1.1<0;1,0>:q    {Compacted}
        add (16|M0)              r13.0<1>:q    r80.0<1;1,0>:q    r1.1<0;1,0>:q    {Compacted}
        send.ugm (32|M0)         r18      r13  null:0  0x0            0x08200B80           {I@1,$0} // wr:4+0, rd:2; load.ugm.d16u32.a64
        add (16|M0)              r9.0<1>:q     r76.0<1;1,0>:q    r1.1<0;1,0>:q    {Compacted}
        send.ugm (32|M0)         r14      r9  null:0  0x0            0x08200B80           {I@1,$15} // wr:4+0, rd:2; load.ugm.d16u32.a64
(W)     cmp (1|M0)    (lt)f2.0   null<1>:f     r4.1<0;1,0>:f     -105.0:f              
(W)     cmp (1|M0)    (gt)f1.0   null<1>:f     r4.1<0;1,0>:f     105.0:f              
(W)     math.exp (1|M0)          r4.6<1>:f     r2.10<0;1,0>:f                  
        sync.nop                             null                             {Compacted,$8.src}
        shl (16|M0)              r8.0<1>:ud    r36.0<2;1,0>:uw   0x10:uw              {$4.dst}
(W)     math.exp (1|M0)          r4.1<1>:f     r2.4<0;1,0>:f                    {F@1}
(W)     mul (1|M0)               r3.12<1>:f    r4.1<0;1,0>:f     r4.6<0;1,0>:f    {M@1}
        mov (16|M16)             r5.0<1>:uw    r37.0<2;1,0>:uw                 
(W&~f2.0) sel (1|M0)             r3.12<1>:f    r3.12<0;1,0>:f    0.0:f               {F@1}
        shl (16|M16)             r11.0<1>:ud   r37.0<2;1,0>:uw   0x10:uw              {$15.src}
(W&~f1.0) sel (1|M0)             r1.10<1>:f    r3.12<0;1,0>:f    inf:f               {F@1}
        mov (16|M0)              r9.0<1>:uw    r36.0<2;1,0>:uw                 
        sync.nop                             null                             {Compacted,$3.src}
        mov (16|M0)              r20.0<1>:uw   r38.0<2;1,0>:uw                  {$5.dst}
        shl (16|M0)              r10.0<1>:ud   r38.0<2;1,0>:uw   0x10:uw             
        mul (32|M0)              acc0.0<1>:f   r68.0<1;1,0>:f    r1.10<0;1,0>:f   {Compacted,F@1}
        mul (16|M0)              acc2.0<1>:f   r9.0<1;1,0>:bf    r8.0<1;1,0>:f    {I@3}
        mul (16|M16)             acc3.0<1>:f   r5.0<1;1,0>:bf    r11.0<1;1,0>:f  
        shl (16|M16)             r16.0<1>:ud   r39.0<2;1,0>:uw   0x10:uw              {$0.src}
        mov (16|M16)             r13.0<1>:uw   r39.0<2;1,0>:uw                 
        mad (16|M0)              acc2.0<1>:f   acc2.0<1;0>:f     r20.0<1;0>:bf     r10.0<1>:f       {I@3}
        sync.nop                             null                             {Compacted,F@3}
        shl (16|M0)              r8.0<1>:ud    r40.0<2;1,0>:uw   0x10:uw              {$6.dst}
        mov (16|M0)              r11.0<1>:uw   r40.0<2;1,0>:uw                  {F@2}
        mad (16|M16)             acc3.0<1>:f   acc3.0<1;0>:f     r13.0<1;0>:bf     r16.0<1>:f       {I@3}
        shl (16|M16)             r12.0<1>:ud   r41.0<2;1,0>:uw   0x10:uw             
        mov (16|M16)             r38.0<1>:uw   r41.0<2;1,0>:uw                 
        mad (16|M0)              acc2.0<1>:f   acc2.0<1;0>:f     r11.0<1;0>:bf     r8.0<1>:f        {I@3}
        shl (16|M16)             r17.0<1>:ud   r43.0<2;1,0>:uw   0x10:uw              {$7.dst}
        shl (16|M0)              r10.0<1>:ud   r42.0<2;1,0>:uw   0x10:uw              {F@3}
        mov (16|M16)             r40.0<1>:uw   r43.0<2;1,0>:uw                 
        mov (16|M0)              r41.0<1>:uw   r42.0<2;1,0>:uw                 
        mad (16|M16)             acc3.0<1>:f   acc3.0<1;0>:f     r38.0<1;0>:bf     r12.0<1>:f       {I@5}
(W)     mov (32|M0)              r46.0<1>:ud   0x0:ud                              {$12.src}
        mov (16|M0)              r39.0<1>:uw   r6.0<2;1,0>:uw                   {$3.dst}
        shl (16|M0)              r8.0<1>:ud    r6.0<2;1,0>:uw    0x10:uw              {F@2}
        mad (16|M0)              r46.0<1>:f    acc2.0<1;0>:f     r41.0<1;0>:bf     r10.0<1>:f       {I@3}
        mad (16|M16)             r47.0<1>:f    acc3.0<1;0>:f     r40.0<1;0>:bf     r17.0<1>:f      
        mov (16|M16)             r10.0<1>:uw   r7.0<2;1,0>:uw                   {F@2}
        shl (16|M16)             r17.0<1>:ud   r7.0<2;1,0>:uw    0x10:uw              {F@1}
        shl (16|M0)              r12.0<1>:ud   r44.0<2;1,0>:uw   0x10:uw              {$8.dst}
        mov (16|M0)              r7.0<1>:uw    r44.0<2;1,0>:uw                 
        mul (16|M0)              acc2.0<1>:f   r39.0<1;1,0>:bf   r8.0<1;1,0>:f    {I@5}
        mul (16|M16)             acc3.0<1>:f   r10.0<1;1,0>:bf   r17.0<1;1,0>:f   {I@3}
        shl (16|M16)             r16.0<1>:ud   r45.0<2;1,0>:uw   0x10:uw             
        mov (16|M16)             r44.0<1>:uw   r45.0<2;1,0>:uw                 
        mad (16|M0)              acc2.0<1>:f   acc2.0<1;0>:f     r7.0<1;0>:bf      r12.0<1>:f       {I@3}
(W)     add (16|M0)              r36.0<1>:f    r46.0<1;1,0>:f    r47.0<1;1,0>:f   {Compacted}
        sync.nop                             null                             {Compacted,F@4}
        shl (16|M0)              r8.0<1>:ud    r18.0<2;1,0>:uw   0x10:uw              {$0.dst}
        mov (16|M0)              r45.0<1>:uw   r18.0<2;1,0>:uw                 
        mad (16|M16)             acc3.0<1>:f   acc3.0<1;0>:f     r44.0<1;0>:bf     r16.0<1>:f       {I@3}
        shl (16|M16)             r17.0<1>:ud   r19.0<2;1,0>:uw   0x10:uw              {F@4}
(W)     mov (8|M0)               r37.0<1>:ud   r36.8<1;1,0>:ud                  {Compacted,F@2}
        mov (16|M16)             r46.0<1>:uw   r19.0<2;1,0>:uw                 
        mad (16|M0)              acc2.0<1>:f   acc2.0<1;0>:f     r45.0<1;0>:bf     r8.0<1>:f        {I@4}
        mov (16|M0)              r80.0<1>:uw   r14.0<2;1,0>:uw                  {$15.dst}
        shl (16|M0)              r6.0<1>:ud    r14.0<2;1,0>:uw   0x10:uw             
        shl (16|M16)             r12.0<1>:ud   r15.0<2;1,0>:uw   0x10:uw             
        mov (16|M16)             r47.0<1>:uw   r15.0<2;1,0>:uw                 
(W)     add (8|M0)               r16.0<1>:f    r36.0<1;1,0>:f    r37.0<1;1,0>:f   {Compacted,I@6}
        mad (16|M16)             acc3.0<1>:f   acc3.0<1;0>:f     r46.0<1;0>:bf     r17.0<1>:f       {I@5}
(W)     mov (32|M0)              r48.0<1>:ud   0x0:ud                              {$0.src}
(W)     mov (4|M0)               r8.0<1>:ud    r16.4<1;1,0>:ud                  {Compacted,F@2}
        mad (16|M0)              r48.0<1>:f    acc2.0<1;0>:f     r80.0<1;0>:bf     r6.0<1>:f        {I@2}
        mad (16|M16)             r49.0<1>:f    acc3.0<1;0>:f     r47.0<1;0>:bf     r12.0<1>:f      
(W)     add (4|M0)               r8.0<1>:f     r16.0<1;1,0>:f    r8.0<1;1,0>:f    {Compacted,I@1}
(W)     add (16|M0)              r6.0<1>:f     r48.0<1;1,0>:f    r49.0<1;1,0>:f   {Compacted,F@2}
(W)     add (1|M0)               r4.8<1>:f     r8.0<0;1,0>:f     r8.2<0;1,0>:f    {F@2}
(W)     add (1|M0)               r4.9<1>:f     r8.1<0;1,0>:f     r8.3<0;1,0>:f   
(W)     mov (8|M0)               r8.0<1>:ud    r6.8<1;1,0>:ud                   {Compacted,F@1}
(W)     add (1|M0)               r4.9<1>:f     r4.8<0;1,0>:f     r4.9<0;1,0>:f   
(W)     add (8|M0)               r6.0<1>:f     r6.0<1;1,0>:f     r8.0<1;1,0>:f    {Compacted,I@1}
(W)     add (1|M0)               r4.13<1>:f    r4.9<0;1,0>:f     1e-06:f               {F@2}
(W)     mov (4|M0)               r8.0<1>:ud    r6.4<1;1,0>:ud                   {Compacted,F@2}
(W)     math.rsqt (1|M0)         r4.1<1>:f     r4.13<0;1,0>:f                   {F@1}
(W)     add (4|M0)               r6.0<1>:f     r6.0<1;1,0>:f     r8.0<1;1,0>:f    {Compacted,I@1}
        mul (16|M0)              r16.0<1>:f    r9.0<1;1,0>:bf    r4.1<0;1,0>:f    {M@1}
(W)     add (1|M0)               r6.4<1>:f     r6.0<0;1,0>:f     r6.2<0;1,0>:f    {Compacted,F@2}
(W)     add (1|M0)               r6.5<1>:f     r6.1<0;1,0>:f     r6.3<0;1,0>:f   
        mul (16|M16)             r17.0<1>:f    r5.0<1;1,0>:bf    r4.1<0;1,0>:f   
(W)     add (1|M0)               r4.8<1>:f     r6.4<0;1,0>:f     r6.5<0;1,0>:f    {F@2}
        mul (32|M0)              r78.0<1>:f    r70.0<1;1,0>:f    r1.10<0;1,0>:f   {Compacted}
(W)     add (1|M0)               r4.12<1>:f    r4.8<0;1,0>:f     1e-06:f               {F@2}
        mul (16|M0)              r36.0<1>:f    r20.0<1;1,0>:bf   r4.1<0;1,0>:f   
        mul (16|M16)             r37.0<1>:f    r13.0<1;1,0>:bf   r4.1<0;1,0>:f   
        mul (16|M0)              r12.0<1>:f    r11.0<1;1,0>:bf   r4.1<0;1,0>:f   
        mul (16|M0)              r8.0<1>:f     r41.0<1;1,0>:bf   r4.1<0;1,0>:f   
        mul (16|M16)             r9.0<1>:f     r40.0<1;1,0>:bf   r4.1<0;1,0>:f   
        mul (16|M16)             r13.0<1>:f    r38.0<1;1,0>:bf   r4.1<0;1,0>:f   
(W)     math.sqt (1|M0)          r4.1<1>:f     r4.12<0;1,0>:f                   {F@1}
        mul (32|M0)              acc0.0<1>:f   acc0.0<1;1,0>:f   r16.0<1;1,0>:f   {Compacted}
        mul (32|M0)              r76.0<1>:f    r66.0<1;1,0>:f    r1.10<0;1,0>:f   {Compacted}
        mad (32|M0)              acc0.0<1>:f   acc0.0<1;0>:f     r78.0<1;0>:f      r36.0<1>:f       {Compacted}
(W)     mul (1|M0)               r4.1<1>:f     r4.7<0;1,0>:f     r4.1<0;1,0>:f    {M@1}
        mad (32|M0)              r42.0<1>:f    acc0.0<1;0>:f     r76.0<1;0>:f      r12.0<1>:f       {Compacted,F@3}
(W)     math.inv (1|M0)          r4.1<1>:f     r4.1<0;1,0>:f                    {F@2}
(W)     mul (1|M0)               acc0.0<1>:d   r2.5<0;1,0>:d     r3.8<0;1,0>:uw   {F@1}
        mul (32|M0)              acc2.0<1>:f   r64.0<1;1,0>:f    r1.10<0;1,0>:f   {Compacted}
(W)     macl (1|M0)              r5.0<1>:d     r2.5<0;1,0>:d     r3.4<0;1,0>:d   
        mul (16|M0)              r18.0<1>:f    r39.0<1;1,0>:bf   r4.1<0;1,0>:f    {M@1}
        mul (16|M16)             r19.0<1>:f    r10.0<1;1,0>:bf   r4.1<0;1,0>:f   
        mul (16|M0)              r38.0<1>:f    r7.0<1;1,0>:bf    r4.1<0;1,0>:f   
        mul (16|M0)              r14.0<1>:f    r45.0<1;1,0>:bf   r4.1<0;1,0>:f   
        mul (16|M16)             r15.0<1>:f    r46.0<1;1,0>:bf   r4.1<0;1,0>:f   
        mul (16|M16)             r11.0<1>:f    r47.0<1;1,0>:bf   r4.1<0;1,0>:f   
        mul (16|M16)             r39.0<1>:f    r44.0<1;1,0>:bf   r4.1<0;1,0>:f   
        mul (16|M0)              r10.0<1>:f    r80.0<1;1,0>:bf   r4.1<0;1,0>:f   
(W)     add (1|M0)               r4.1<1>:d     r5.0<0;1,0>:d     r3.6<0;1,0>:d    {Compacted,A@1}
(W)     mov (32|M0)              r40.0<1>:ud   0x0:ud                             
(W)     shl (1|M0)               r2.2<1>:q     r4.1<0;1,0>:d     1:w               {I@2}
        mad (32|M0)              r40.0<1>:f    r42.0<1;0>:f      acc2.0<1;0>:f     r8.0<1>:f        {Compacted,I@2}
(W)     mov (2|M0)               r4.16<1>:w    0x40:uv                             
(W)     add (1|M0)               r46.0<1>:q    r2.2<0;1,0>:q     r1.3<0;1,0>:q    {Compacted,I@2}
(W)     add (16|M0)              r5.0<1>:f     r40.0<1;1,0>:f    r41.0<1;1,0>:f   {Compacted,F@1}
(W)     add (1|M0)               r20.0<1>:uq   r46.0<0;1,0>:uq   r4.16<0;1,0>:w   {I@1}
(W)     add (1|M0)               r20.1<1>:uq   r46.0<0;1,0>:uq   r4.17<0;1,0>:w  
(W)     mov (8|M0)               r47.0<1>:ud   r5.8<1;1,0>:ud                   {Compacted,F@1}
        mul (32|M0)              acc0.0<1>:f   r60.0<1;1,0>:f    r1.10<0;1,0>:f   {Compacted}
(W)     add (8|M0)               r46.0<1>:f    r5.0<1;1,0>:f     r47.0<1;1,0>:f   {Compacted,I@1}
(W)     send.ugm (2|M0)          r5       r20  null:0  0x0            0x04100580           {F@1,$4} // wr:2+0, rd:1; load.ugm.d32.a64
        mul (32|M0)              acc2.0<1>:f   r62.0<1;1,0>:f    r1.10<0;1,0>:f   {Compacted}
        mul (32|M0)              acc0.0<1>:f   acc0.0<1;1,0>:f   r16.0<1;1,0>:f   {Compacted}
        mul (32|M0)              r42.0<1>:f    r58.0<1;1,0>:f    r1.10<0;1,0>:f   {Compacted}
        mad (32|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc2.0<1;0>:f     r36.0<1>:f       {Compacted}
        mul (32|M0)              acc2.0<1>:f   r50.0<1;1,0>:f    r1.10<0;1,0>:f   {Compacted}
        mad (32|M0)              acc0.0<1>:f   acc0.0<1;0>:f     r42.0<1;0>:f      r12.0<1>:f       {Compacted,F@3}
        mul (32|M0)              r44.0<1>:f    r56.0<1;1,0>:f    r1.10<0;1,0>:f   {Compacted}
        mul (32|M0)              acc2.0<1>:f   acc2.0<1;1,0>:f   r16.0<1;1,0>:f   {Compacted}
(W)     mov (32|M0)              r40.0<1>:ud   0x0:ud                             
        mul (32|M0)              r42.0<1>:f    r54.0<1;1,0>:f    r1.10<0;1,0>:f   {Compacted}
        mad (32|M0)              r40.0<1>:f    acc0.0<1;0>:f     r44.0<1;0>:f      r8.0<1>:f        {Compacted,A@1}
        mul (32|M0)              acc0.0<1>:f   r34.0<1;1,0>:f    r1.10<0;1,0>:f   {Compacted}
(W)     mov (4|M0)               r20.0<1>:ud   r46.4<1;1,0>:ud                  {Compacted,$4.src}
        mad (32|M0)              acc2.0<1>:f   acc2.0<1;0>:f     r42.0<1;0>:f      r36.0<1>:f       {Compacted,F@3}
(W)     add (4|M0)               r20.0<1>:f    r46.0<1;1,0>:f    r20.0<1;1,0>:f   {Compacted,I@1}
        mul (32|M0)              r42.0<1>:f    r32.0<1;1,0>:f    r1.10<0;1,0>:f   {Compacted}
        mad (32|M0)              acc0.0<1>:f   acc2.0<1;0>:f     acc0.0<1;0>:f     r12.0<1>:f       {Compacted}
        mul (32|M0)              acc2.0<1>:f   r28.0<1;1,0>:f    r1.10<0;1,0>:f   {Compacted}
(W)     add (1|M0)               r4.8<1>:f     r20.0<0;1,0>:f    r20.2<0;1,0>:f   {F@4}
(W)     add (1|M0)               r4.9<1>:f     r20.1<0;1,0>:f    r20.3<0;1,0>:f  
(W)     add (16|M0)              r20.0<1>:f    r40.0<1;1,0>:f    r41.0<1;1,0>:f   {Compacted}
(W)     mov (32|M0)              r40.0<1>:ud   0x0:ud                              {F@1}
        mad (32|M0)              r40.0<1>:f    acc0.0<1;0>:f     r42.0<1;0>:f      r8.0<1>:f        {Compacted,I@1}
        mul (32|M0)              r44.0<1>:f    r30.0<1;1,0>:f    r1.10<0;1,0>:f   {Compacted}
        mul (32|M0)              acc0.0<1>:f   acc2.0<1;1,0>:f   r16.0<1;1,0>:f   {Compacted}
(W)     mov (8|M0)               r46.0<1>:ud   r20.8<1;1,0>:ud                  {Compacted}
        mul (32|M0)              acc2.0<1>:f   r26.0<1;1,0>:f    r1.10<0;1,0>:f   {Compacted}
(W)     add (8|M0)               r46.0<1>:f    r20.0<1;1,0>:f    r46.0<1;1,0>:f   {Compacted,I@1}
        mad (32|M0)              acc0.0<1>:f   acc0.0<1;0>:f     r44.0<1;0>:f      r36.0<1>:f       {Compacted,F@4}
        mul (32|M0)              r42.0<1>:f    r24.0<1;1,0>:f    r1.10<0;1,0>:f   {Compacted}
(W)     add (16|M0)              r20.0<1>:f    r40.0<1;1,0>:f    r41.0<1;1,0>:f   {Compacted}
(W)     mov (4|M0)               r47.0<1>:ud   r46.4<1;1,0>:ud                  {Compacted,F@4}
        mad (32|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc2.0<1;0>:f     r12.0<1>:f       {Compacted}
(W)     mov (32|M0)              r40.0<1>:ud   0x0:ud                              {F@2}
(W)     mov (8|M0)               r45.0<1>:ud   r20.8<1;1,0>:ud                  {Compacted}
(W)     add (4|M0)               r44.0<1>:f    r46.0<1;1,0>:f    r47.0<1;1,0>:f   {Compacted,I@3}
        mad (32|M0)              r40.0<1>:f    acc0.0<1;0>:f     r42.0<1;0>:f      r8.0<1>:f        {Compacted,I@2}
(W)     add (1|M0)               r4.1<1>:f     r4.8<0;1,0>:f     r4.9<0;1,0>:f   
(W)     add (1|M0)               r4.8<1>:f     r44.0<0;1,0>:f    r44.2<0;1,0>:f   {F@3}
(W)     add (1|M0)               r4.9<1>:f     r44.1<0;1,0>:f    r44.3<0;1,0>:f  
(W)     add (8|M0)               r44.0<1>:f    r20.0<1;1,0>:f    r45.0<1;1,0>:f   {Compacted,I@1}
(W)     add (16|M0)              r20.0<1>:f    r40.0<1;1,0>:f    r41.0<1;1,0>:f   {Compacted,F@5}
(W)     mov (4|M0)               r42.0<1>:ud   r44.4<1;1,0>:ud                  {Compacted,F@2}
(W)     mov (8|M0)               r41.0<1>:ud   r20.8<1;1,0>:ud                  {Compacted,F@1}
(W)     add (4|M0)               r40.0<1>:f    r44.0<1;1,0>:f    r42.0<1;1,0>:f   {Compacted,I@2}
(W)     add (8|M0)               r20.0<1>:f    r20.0<1;1,0>:f    r41.0<1;1,0>:f   {Compacted,I@1}
(W)     add (1|M0)               r4.6<1>:f     r4.8<0;1,0>:f     r4.9<0;1,0>:f   
(W)     add (1|M0)               r4.8<1>:f     r40.0<0;1,0>:f    r40.2<0;1,0>:f   {F@3}
(W)     add (1|M0)               r4.9<1>:f     r40.1<0;1,0>:f    r40.3<0;1,0>:f  
(W)     mov (4|M0)               r40.0<1>:ud   r20.4<1;1,0>:ud                  {Compacted,F@1}
(W)     add (1|M0)               r4.8<1>:f     r4.8<0;1,0>:f     r4.9<0;1,0>:f   
(W)     add (4|M0)               r20.0<1>:f    r20.0<1;1,0>:f    r40.0<1;1,0>:f   {Compacted,I@1}
        sync.nop                             null                             {Compacted,F@2}
(W)     shl (1|M0)               r4.9<1>:ud    r5.0<0;1,0>:uw    0x10:uw              {$4.dst}
(W)     add (1|M0)               r4.11<1>:f    r20.0<0;1,0>:f    r20.2<0;1,0>:f   {F@1}
(W)     add (1|M0)               r4.12<1>:f    r20.1<0;1,0>:f    r20.3<0;1,0>:f  
        sync.nop                             null                             {Compacted,$10.src}
(W)     shr (1|M0)               a0.0<1>:ud    r21.5<0;1,0>:ud   0x4:uw              {$2.src}
(W)     add (1|M0)               r5.2<1>:f     r4.11<0;1,0>:f    r4.12<0;1,0>:f   {F@1}
(W)     shl (1|M0)               r4.12<1>:ud   r5.1<0;1,0>:uw    0x10:uw              {F@1}
(W)     add (1|M0)               r4.11<1>:f    r4.9<0;1,0>:f     -r4.1<0;1,0>:f   {I@3}
(W)     add (1|M0)               r4.1<1>:f     r4.12<0;1,0>:f    -r4.6<0;1,0>:f   {I@1}
(W)     mul (1|M0)               r4.6<1>:f     r4.11<0;1,0>:f    r2.6<0;1,0>:f    {F@2}
        shl (32|M0)              r6.0<1>:d     r108.0<1;1,0>:d   2:w               {Compacted}
        mul (32|M0)              r40.0<1>:f    r36.0<1;1,0>:f    r4.6<0;1,0>:f    {Compacted,F@1}
        send.ugm (32|M0)         null     r6  r40:2  a0.0        0x440E0504           {A@1,$2} // wr:2+2, rd:0; store.ugm.d32.a32.wb.wb.ss[a0.0]
        mul (32|M0)              r42.0<1>:f    r16.0<1;1,0>:f    r4.6<0;1,0>:f    {Compacted}
        mul (32|M0)              r46.0<1>:f    r12.0<1;1,0>:f    r4.6<0;1,0>:f    {Compacted}
        mul (32|M0)              r76.0<1>:f    r8.0<1;1,0>:f     r4.6<0;1,0>:f    {Compacted}
(W)     mul (1|M0)               r4.1<1>:f     r4.1<0;1,0>:f     r2.6<0;1,0>:f   
(W)     shl (1|M0)               r4.15<1>:ud   r5.2<0;1,0>:uw    0x10:uw             
        mul (32|M0)              r44.0<1>:f    r36.0<1;1,0>:f    r4.1<0;1,0>:f    {Compacted,F@1}
        mul (32|M0)              r48.0<1>:f    r16.0<1;1,0>:f    r4.1<0;1,0>:f    {Compacted}
        mul (32|M0)              r78.0<1>:f    r12.0<1;1,0>:f    r4.1<0;1,0>:f    {Compacted}
        mul (32|M0)              r80.0<1>:f    r8.0<1;1,0>:f     r4.1<0;1,0>:f    {Compacted}
(W)     add (1|M0)               r4.9<1>:f     r4.15<0;1,0>:f    -r4.8<0;1,0>:f   {I@1}
(W)     shl (1|M0)               r4.13<1>:ud   r5.3<0;1,0>:uw    0x10:uw             
(W)     mul (1|M0)               r1.11<1>:f    r4.9<0;1,0>:f     r2.6<0;1,0>:f    {F@1}
(W)     add (1|M0)               r4.8<1>:f     r4.13<0;1,0>:f    -r5.2<0;1,0>:f   {I@1}
(W)     mul (1|M0)               r1.14<1>:f    r4.8<0;1,0>:f     r2.6<0;1,0>:f    {F@1}
        send.ugm (32|M0)         r40      r6  null:0  a0.0        0x44280500           {$5} // wr:2+0, rd:2; load.ugm.d32.a32.ca.ca.ss[a0.0]
        send.ugm (32|M0)         null     r6  r42:2  a0.0        0x440E0504           {$6} // wr:2+2, rd:0; store.ugm.d32.a32.wb.wb.ss[a0.0]
        mul (32|M0)              r42.0<1>:f    r16.0<1;1,0>:f    r1.11<0;1,0>:f   {Compacted,$6.src}
        mad (32|M0)              r70.0<1>:f    r40.0<1;0>:f      r70.0<1;0>:f      r1.10<0>:f       {Compacted,$5.dst}
        send.ugm (32|M0)         r40      r6  null:0  a0.0        0x44280500           {F@1,$7} // wr:2+0, rd:2; load.ugm.d32.a32.ca.ca.ss[a0.0]
        send.ugm (32|M0)         null     r6  r46:2  a0.0        0x440E0504           {$8} // wr:2+2, rd:0; store.ugm.d32.a32.wb.wb.ss[a0.0]
        mul (32|M0)              r46.0<1>:f    r8.0<1;1,0>:f     r1.11<0;1,0>:f   {Compacted,$8.src}
        mad (32|M0)              r68.0<1>:f    r40.0<1;0>:f      r68.0<1;0>:f      r1.10<0>:f       {Compacted,$7.dst}
        mul (32|M0)              acc0.0<1>:f   r68.0<1;1,0>:f    r18.0<1;1,0>:f   {Compacted,F@1}
        mad (32|M0)              acc0.0<1>:f   acc0.0<1;0>:f     r70.0<1;0>:f      r38.0<1>:f       {Compacted}
        send.ugm (32|M0)         r40      r6  null:0  a0.0        0x44280500           {$3} // wr:2+0, rd:2; load.ugm.d32.a32.ca.ca.ss[a0.0]
        send.ugm (32|M0)         null     r6  r76:2  a0.0        0x440E0504           {$4} // wr:2+2, rd:0; store.ugm.d32.a32.wb.wb.ss[a0.0]
        mul (32|M0)              r76.0<1>:f    r12.0<1;1,0>:f    r1.14<0;1,0>:f   {Compacted,$4.src}
        mad (32|M0)              r66.0<1>:f    r40.0<1;0>:f      r66.0<1;0>:f      r1.10<0>:f       {Compacted,$3.dst}
        mad (32|M0)              acc0.0<1>:f   acc0.0<1;0>:f     r66.0<1;0>:f      r14.0<1>:f       {Compacted,F@1}
        send.ugm (32|M0)         r40      r6  null:0  a0.0        0x44280500           {$1} // wr:2+0, rd:2; load.ugm.d32.a32.ca.ca.ss[a0.0]
        send.ugm (32|M0)         null     r6  r44:2  a0.0        0x440E0504           {$0} // wr:2+2, rd:0; store.ugm.d32.a32.wb.wb.ss[a0.0]
        mul (32|M0)              r44.0<1>:f    r12.0<1;1,0>:f    r1.11<0;1,0>:f   {Compacted,$0.src}
        mad (32|M0)              r64.0<1>:f    r40.0<1;0>:f      r64.0<1;0>:f      r1.10<0>:f       {Compacted,$1.dst}
(W)     mov (32|M0)              r12.0<1>:ud   0x0:ud                              {F@2}
        send.ugm (32|M0)         r40      r6  null:0  a0.0        0x44280500           {F@1,$15} // wr:2+0, rd:2; load.ugm.d32.a32.ca.ca.ss[a0.0]
        send.ugm (32|M0)         null     r6  r48:2  a0.0        0x440E0504           {$14} // wr:2+2, rd:0; store.ugm.d32.a32.wb.wb.ss[a0.0]
        mul (32|M0)              r48.0<1>:f    r8.0<1;1,0>:f     r1.14<0;1,0>:f   {Compacted,$14.src}
        mad (32|M0)              r62.0<1>:f    r40.0<1;0>:f      r62.0<1;0>:f      r1.10<0>:f       {Compacted,$15.dst}
(W)     mov (32|M0)              r8.0<1>:ud    0x0:ud                              {F@2}
        send.ugm (32|M0)         r40      r6  null:0  a0.0        0x44280500           {F@1,$2} // wr:2+0, rd:2; load.ugm.d32.a32.ca.ca.ss[a0.0]
        send.ugm (32|M0)         null     r6  r78:2  a0.0        0x440E0504           {$13} // wr:2+2, rd:0; store.ugm.d32.a32.wb.wb.ss[a0.0]
        mul (32|M0)              r78.0<1>:f    r16.0<1;1,0>:f    r1.14<0;1,0>:f   {Compacted,$13.src}
        mad (32|M0)              r60.0<1>:f    r40.0<1;0>:f      r60.0<1;0>:f      r1.10<0>:f       {Compacted,$2.dst}
        mul (32|M0)              acc2.0<1>:f   r60.0<1;1,0>:f    r18.0<1;1,0>:f   {Compacted,F@1}
        mad (32|M0)              acc2.0<1>:f   acc2.0<1;0>:f     r62.0<1;0>:f      r38.0<1>:f       {Compacted}
(W)     mov (32|M0)              r16.0<1>:ud   0x0:ud                             
        send.ugm (32|M0)         r40      r6  null:0  a0.0        0x44280500           {$6} // wr:2+0, rd:2; load.ugm.d32.a32.ca.ca.ss[a0.0]
        send.ugm (32|M0)         null     r6  r80:2  a0.0        0x440E0504           {$9} // wr:2+2, rd:0; store.ugm.d32.a32.wb.wb.ss[a0.0]
        mad (32|M0)              r58.0<1>:f    r40.0<1;0>:f      r58.0<1;0>:f      r1.10<0>:f       {Compacted,$6.dst}
        mad (32|M0)              acc2.0<1>:f   acc2.0<1;0>:f     r58.0<1;0>:f      r14.0<1>:f       {Compacted,F@1}
        send.ugm (32|M0)         r40      r6  null:0  a0.0        0x44280500           {$5} // wr:2+0, rd:2; load.ugm.d32.a32.ca.ca.ss[a0.0]
        mad (32|M0)              r56.0<1>:f    r40.0<1;0>:f      r56.0<1;0>:f      r1.10<0>:f       {Compacted,$5.dst}
        mul (32|M0)              r40.0<1>:f    r36.0<1;1,0>:f    r1.11<0;1,0>:f   {Compacted}
        send.ugm (32|M0)         null     r6  r40:2  a0.0        0x440E0504           {F@1,$8} // wr:2+2, rd:0; store.ugm.d32.a32.wb.wb.ss[a0.0]
        mad (32|M0)              r16.0<1>:f    acc2.0<1;0>:f     r56.0<1;0>:f      r10.0<1>:f       {Compacted,I@1}
(W)     add (16|M0)              r17.0<1>:f    r16.0<1;1,0>:f    r17.0<1;1,0>:f   {Compacted,F@1}
(W)     mov (8|M0)               r16.0<1>:ud   r17.8<1;1,0>:ud                  {Compacted,F@1}
        mul (32|M0)              r40.0<1>:f    r36.0<1;1,0>:f    r1.14<0;1,0>:f   {Compacted,$8.src}
(W)     add (8|M0)               r17.0<1>:f    r17.0<1;1,0>:f    r16.0<1;1,0>:f   {Compacted,I@1}
(W)     mov (32|M0)              r36.0<1>:ud   0x0:ud                              {F@2}
(W)     mov (4|M0)               r16.0<1>:ud   r17.4<1;1,0>:ud                  {Compacted,F@1}
        mad (32|M0)              r36.0<1>:f    acc0.0<1;0>:f     r64.0<1;0>:f      r10.0<1>:f       {Compacted,I@2}
(W)     add (4|M0)               r16.0<1>:f    r17.0<1;1,0>:f    r16.0<1;1,0>:f   {Compacted,I@1}
(W)     add (16|M0)              r5.0<1>:f     r36.0<1;1,0>:f    r37.0<1;1,0>:f   {Compacted,F@2}
(W)     add (1|M0)               r4.8<1>:f     r16.0<0;1,0>:f    r16.2<0;1,0>:f   {F@2}
(W)     add (1|M0)               r4.9<1>:f     r16.1<0;1,0>:f    r16.3<0;1,0>:f  
(W)     mov (8|M0)               r20.0<1>:ud   r5.8<1;1,0>:ud                   {Compacted,F@3}
(W)     add (1|M0)               r2.6<1>:f     r4.8<0;1,0>:f     r4.9<0;1,0>:f    {F@1}
(W)     add (8|M0)               r5.0<1>:f     r5.0<1;1,0>:f     r20.0<1;1,0>:f   {Compacted,I@1}
(W)     mov (4|M0)               r20.0<1>:ud   r5.4<1;1,0>:ud                   {Compacted,F@1}
(W)     add (4|M0)               r5.0<1>:f     r5.0<1;1,0>:f     r20.0<1;1,0>:f   {Compacted,I@1}
(W)     add (1|M0)               r4.11<1>:f    r5.0<0;1,0>:f     r5.2<0;1,0>:f    {F@1}
(W)     add (1|M0)               r4.12<1>:f    r5.1<0;1,0>:f     r5.3<0;1,0>:f   
(W)     add (1|M0)               r2.10<1>:f    r4.11<0;1,0>:f    r4.12<0;1,0>:f   {F@1}
        send.ugm (32|M0)         r16      r6  null:0  a0.0        0x44280500           {$7} // wr:2+0, rd:2; load.ugm.d32.a32.ca.ca.ss[a0.0]
        send.ugm (32|M0)         null     r6  r42:2  a0.0        0x440E0504           {$10} // wr:2+2, rd:0; store.ugm.d32.a32.wb.wb.ss[a0.0]
        mad (32|M0)              r54.0<1>:f    r16.0<1;0>:f      r54.0<1;0>:f      r1.10<0>:f       {Compacted,$7.dst}
        send.ugm (32|M0)         r16      r6  null:0  a0.0        0x44280500           {F@1,$4} // wr:2+0, rd:2; load.ugm.d32.a32.ca.ca.ss[a0.0]
        send.ugm (32|M0)         null     r6  r44:2  a0.0        0x440E0504           {$11} // wr:2+2, rd:0; store.ugm.d32.a32.wb.wb.ss[a0.0]
        mad (32|M0)              r50.0<1>:f    r16.0<1;0>:f      r50.0<1;0>:f      r1.10<0>:f       {Compacted,$4.dst}
        mul (32|M0)              acc0.0<1>:f   r50.0<1;1,0>:f    r18.0<1;1,0>:f   {Compacted,F@1}
        mad (32|M0)              acc0.0<1>:f   acc0.0<1;0>:f     r54.0<1;0>:f      r38.0<1>:f       {Compacted}
        send.ugm (32|M0)         r16      r6  null:0  a0.0        0x44280500           {$5} // wr:2+0, rd:2; load.ugm.d32.a32.ca.ca.ss[a0.0]
        send.ugm (32|M0)         null     r6  r46:2  a0.0        0x440E0504           {$12} // wr:2+2, rd:0; store.ugm.d32.a32.wb.wb.ss[a0.0]
        mad (32|M0)              r34.0<1>:f    r16.0<1;0>:f      r34.0<1;0>:f      r1.10<0>:f       {Compacted,$5.dst}
        mad (32|M0)              acc0.0<1>:f   acc0.0<1;0>:f     r34.0<1;0>:f      r14.0<1>:f       {Compacted,F@1}
        send.ugm (32|M0)         r16      r6  null:0  a0.0        0x44280500           {$6} // wr:2+0, rd:2; load.ugm.d32.a32.ca.ca.ss[a0.0]
        send.ugm (32|M0)         null     r6  r40:2  a0.0        0x440E0504           {$13} // wr:2+2, rd:0; store.ugm.d32.a32.wb.wb.ss[a0.0]
        mad (32|M0)              r32.0<1>:f    r16.0<1;0>:f      r32.0<1;0>:f      r1.10<0>:f       {Compacted,$6.dst}
        mad (32|M0)              r12.0<1>:f    acc0.0<1;0>:f     r32.0<1;0>:f      r10.0<1>:f       {Compacted,F@1}
(W)     add (16|M0)              r5.0<1>:f     r12.0<1;1,0>:f    r13.0<1;1,0>:f   {Compacted,F@1}
(W)     mov (8|M0)               r12.0<1>:ud   r5.8<1;1,0>:ud                   {Compacted,F@1}
(W)     add (8|M0)               r5.0<1>:f     r5.0<1;1,0>:f     r12.0<1;1,0>:f   {Compacted,I@1}
(W)     mov (4|M0)               r12.0<1>:ud   r5.4<1;1,0>:ud                   {Compacted,F@1}
(W)     add (4|M0)               r5.0<1>:f     r5.0<1;1,0>:f     r12.0<1;1,0>:f   {Compacted,I@1}
(W)     add (1|M0)               r4.8<1>:f     r5.0<0;1,0>:f     r5.2<0;1,0>:f    {F@1}
(W)     add (1|M0)               r4.9<1>:f     r5.1<0;1,0>:f     r5.3<0;1,0>:f   
(W)     add (1|M0)               r4.1<1>:f     r4.8<0;1,0>:f     r4.9<0;1,0>:f    {F@1}
        send.ugm (32|M0)         r12      r6  null:0  a0.0        0x44280500           {$3} // wr:2+0, rd:2; load.ugm.d32.a32.ca.ca.ss[a0.0]
        send.ugm (32|M0)         null     r6  r78:2  a0.0        0x440E0504           {$14} // wr:2+2, rd:0; store.ugm.d32.a32.wb.wb.ss[a0.0]
        mad (32|M0)              r30.0<1>:f    r12.0<1;0>:f      r30.0<1;0>:f      r1.10<0>:f       {Compacted,$3.dst}
        send.ugm (32|M0)         r12      r6  null:0  a0.0        0x44280500           {F@1,$1} // wr:2+0, rd:2; load.ugm.d32.a32.ca.ca.ss[a0.0]
        send.ugm (32|M0)         null     r6  r76:2  a0.0        0x440E0504           {$15} // wr:2+2, rd:0; store.ugm.d32.a32.wb.wb.ss[a0.0]
        mad (32|M0)              r28.0<1>:f    r12.0<1;0>:f      r28.0<1;0>:f      r1.10<0>:f       {Compacted,$1.dst}
        mul (32|M0)              acc0.0<1>:f   r28.0<1;1,0>:f    r18.0<1;1,0>:f   {Compacted,F@1}
        mad (32|M0)              acc0.0<1>:f   acc0.0<1;0>:f     r30.0<1;0>:f      r38.0<1>:f       {Compacted}
        send.ugm (32|M0)         r12      r6  null:0  a0.0        0x44280500           {$8} // wr:2+0, rd:2; load.ugm.d32.a32.ca.ca.ss[a0.0]
        send.ugm (32|M0)         null     r6  r48:2  a0.0        0x440E0504           {$0} // wr:2+2, rd:0; store.ugm.d32.a32.wb.wb.ss[a0.0]
        mad (32|M0)              r26.0<1>:f    r12.0<1;0>:f      r26.0<1;0>:f      r1.10<0>:f       {Compacted,$8.dst}
        mad (32|M0)              acc0.0<1>:f   acc0.0<1;0>:f     r26.0<1;0>:f      r14.0<1>:f       {Compacted,F@1}
        send.ugm (32|M0)         r6       r6  null:0  a0.0        0x44280500           {$3} // wr:2+0, rd:2; load.ugm.d32.a32.ca.ca.ss[a0.0]
        mad (32|M0)              r24.0<1>:f    r6.0<1;0>:f       r24.0<1;0>:f      r1.10<0>:f       {Compacted,$3.dst}
        mad (32|M0)              r8.0<1>:f     acc0.0<1;0>:f     r24.0<1;0>:f      r10.0<1>:f       {Compacted,F@1}
(W)     add (16|M0)              r5.0<1>:f     r8.0<1;1,0>:f     r9.0<1;1,0>:f    {Compacted,F@1}
(W)     mov (8|M0)               r6.0<1>:ud    r5.8<1;1,0>:ud                   {Compacted,F@1}
(W)     add (8|M0)               r5.0<1>:f     r5.0<1;1,0>:f     r6.0<1;1,0>:f    {Compacted,I@1}
(W)     mov (4|M0)               r6.0<1>:ud    r5.4<1;1,0>:ud                   {Compacted,F@1}
(W)     add (4|M0)               r5.0<1>:f     r5.0<1;1,0>:f     r6.0<1;1,0>:f    {Compacted,I@1}
(W)     add (1|M0)               r4.8<1>:f     r5.0<0;1,0>:f     r5.2<0;1,0>:f    {F@1}
(W)     add (1|M0)               r4.9<1>:f     r5.1<0;1,0>:f     r5.3<0;1,0>:f   
(W)     add (1|M0)               r4.6<1>:f     r4.8<0;1,0>:f     r4.9<0;1,0>:f    {F@1}
(~f3.0) goto (32|M0)                         L11344                  L11344                
L11208:
(W)     add (1|M0)               r10.0<1>:q    r2.2<0;1,0>:q     r1.0<0;1,0>:q    {Compacted}
(W)     mov (2|M0)               r4.16<1>:w    0x40:uv                              {F@1}
(W)     mov (1|M0)               r8.0<1>:bf    r2.10<0;1,0>:f                  
(W)     mov (1|M0)               r8.1<1>:bf    r2.6<0;1,0>:f                   
(W)     mov (1|M0)               r8.2<1>:bf    r4.1<0;1,0>:f                   
(W)     mov (1|M0)               r8.3<1>:bf    r4.6<0;1,0>:f                   
(W)     add (1|M0)               r6.0<1>:uq    r10.0<0;1,0>:uq   r4.16<0;1,0>:w   {I@1}
(W)     add (1|M0)               r6.1<1>:uq    r10.0<0;1,0>:uq   r4.17<0;1,0>:w  
(W)     send.ugm (2|M0)          null     r6  r8:1  0x0            0x04000584           {A@1,$4} // wr:2+1, rd:0; store.ugm.d32.a64
L11344:
        join (32|M0)                         L11360                                
L11360:
(W)     add3 (1|M0)              r4.1<1>:d     r2.14<0;0>:d      r1.15<0;0>:d      -r86.0<0>:d     
(W)     shl (1|M0)               r4.4<1>:q     r4.1<0;1,0>:d     2:w               {I@1}
(W)     add (1|M0)               r6.0<1>:q     r4.4<0;1,0>:q     r2.6<0;1,0>:q    {@1,$4.src}
(W)     send.ugm (1|M0)          r5       r6  null:0  0x0            0x02108580           {I@1,$5} // wr:1+0, rd:1; load.ugm.d32x1t.a64
(W)     cmp (32|M0)   (gt)f0.0   null<1>:d     r5.0<0;1,0>:d     0:w               {$5.dst}
(W&~f0.0) jmpi                               L13968                                
L11456:
(W)     mul (1|M0)               acc0.0<1>:ud  r1.12<0;1,0>:ud   r5.0<0;1,0>:uw  
(W)     macl (1|M0)              r8.0<1>:ud    r1.12<0;1,0>:ud   r5.0<0;1,0>:ud   {Compacted}
(W)     mul (1|M0)               acc0.0<1>:ud  r1.12<0;1,0>:ud   r5.0<0;1,0>:uw  
(W)     mach (1|M0)              r6.0<1>:d     r1.12<0;1,0>:ud   r5.0<0;1,0>:ud  
(W)     mul (1|M0)               acc0.0<1>:d   r5.0<0;1,0>:ud    r1.26<0;1,0>:uw 
(W)     macl (1|M0)              r5.0<1>:d     r5.0<0;1,0>:ud    r1.13<0;1,0>:d  
(W)     mul (1|M0)               acc0.0<1>:d   r3.6<0;1,0>:d     r3.4<0;1,0>:uw  
(W)     add (1|M0)               r8.1<1>:d     r6.0<0;1,0>:d     r5.0<0;1,0>:d    {Compacted,I@2}
(W)     macl (1|M0)              r5.0<1>:d     r3.6<0;1,0>:d     r3.2<0;1,0>:d    {Compacted}
(W)     shl (1|M0)               r4.4<1>:q     r8.0<0;1,0>:q     1:w               {I@2}
(W)     add (1|M0)               r2.4<1>:d     r2.7<0;1,0>:d     r5.0<0;1,0>:d    {I@2}
        mov (16|M0)              r15.0<1>:bf   r70.0<1;1,0>:f                  
        add (32|M0)              r6.0<1>:d     r22.0<1;1,0>:d    r2.4<0;1,0>:d    {Compacted,I@1}
        mov (16|M0)              r11.0<2>:ud   r6.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M16)             r13.0<2>:ud   r7.0<1;1,0>:ud                   {Compacted}
        mov (16|M16)             r15.16<1>:bf  r71.0<1;1,0>:f                  
(W)     add (1|M0)               r1.5<1>:q     r4.4<0;1,0>:q     r2.4<0;1,0>:q   
        shl (16|M0)              r9.0<1>:q     r11.0<2;1,0>:d    1:w               {I@3}
        shl (16|M16)             r11.0<1>:q    r13.0<2;1,0>:d    1:w               {I@3}
        add (16|M0)              r5.0<1>:q     r1.5<0;1,0>:q     r9.0<1;1,0>:q    {Compacted,I@2}
        add (16|M16)             r7.0<1>:q     r1.5<0;1,0>:q     r11.0<1;1,0>:q   {Compacted,I@2}
        mov (32|M0)              r10.0<1>:ud   r15.0<1;1,0>:uw                  {F@1}
        send.ugm (32|M0)         null     r5  r10:2  0x0            0x08000B84           {I@1,$6} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r6.0<1>:d     r52.0<1;1,0>:d    r2.4<0;1,0>:d    {Compacted,$6.src}
        mov (16|M0)              r11.0<2>:ud   r6.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M16)             r13.0<2>:ud   r7.0<1;1,0>:ud                   {Compacted}
        mov (16|M0)              r15.0<1>:bf   r68.0<1;1,0>:f                  
        mov (16|M16)             r15.16<1>:bf  r69.0<1;1,0>:f                  
        shl (16|M0)              r9.0<1>:q     r11.0<2;1,0>:d    1:w               {I@2}
        shl (16|M16)             r11.0<1>:q    r13.0<2;1,0>:d    1:w               {I@2}
        add (16|M0)              r5.0<1>:q     r1.5<0;1,0>:q     r9.0<1;1,0>:q    {Compacted,I@2}
        add (16|M16)             r7.0<1>:q     r1.5<0;1,0>:q     r11.0<1;1,0>:q   {Compacted,I@2}
        mov (32|M0)              r10.0<1>:ud   r15.0<1;1,0>:uw                  {F@1}
        send.ugm (32|M0)         null     r5  r10:2  0x0            0x08000B84           {I@1,$7} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r6.0<1>:d     r72.0<1;1,0>:d    r2.4<0;1,0>:d    {Compacted,$7.src}
        mov (16|M0)              r11.0<2>:ud   r6.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M16)             r13.0<2>:ud   r7.0<1;1,0>:ud                   {Compacted}
        mov (16|M0)              r15.0<1>:bf   r66.0<1;1,0>:f                  
        mov (16|M16)             r15.16<1>:bf  r67.0<1;1,0>:f                  
        shl (16|M0)              r9.0<1>:q     r11.0<2;1,0>:d    1:w               {I@2}
        shl (16|M16)             r11.0<1>:q    r13.0<2;1,0>:d    1:w               {I@2}
        add (16|M0)              r5.0<1>:q     r1.5<0;1,0>:q     r9.0<1;1,0>:q    {Compacted,I@2}
        add (16|M16)             r7.0<1>:q     r1.5<0;1,0>:q     r11.0<1;1,0>:q   {Compacted,I@2}
        mov (32|M0)              r10.0<1>:ud   r15.0<1;1,0>:uw                  {F@1}
        send.ugm (32|M0)         null     r5  r10:2  0x0            0x08000B84           {I@1,$8} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r6.0<1>:d     r74.0<1;1,0>:d    r2.4<0;1,0>:d    {Compacted,$8.src}
        mov (16|M0)              r11.0<2>:ud   r6.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M16)             r13.0<2>:ud   r7.0<1;1,0>:ud                   {Compacted}
        mov (16|M0)              r15.0<1>:bf   r64.0<1;1,0>:f                  
        mov (16|M16)             r15.16<1>:bf  r65.0<1;1,0>:f                  
        shl (16|M0)              r9.0<1>:q     r11.0<2;1,0>:d    1:w               {I@2}
        shl (16|M16)             r11.0<1>:q    r13.0<2;1,0>:d    1:w               {I@2}
(W)     or (1|M0)                r4.1<1>:d     r3.6<0;1,0>:d     1:w              
        add (16|M0)              r5.0<1>:q     r1.5<0;1,0>:q     r9.0<1;1,0>:q    {Compacted,I@3}
        add (16|M16)             r7.0<1>:q     r1.5<0;1,0>:q     r11.0<1;1,0>:q   {Compacted,I@3}
        mov (32|M0)              r10.0<1>:ud   r15.0<1;1,0>:uw                  {F@1}
        send.ugm (32|M0)         null     r5  r10:2  0x0            0x08000B84           {I@1,$3} // wr:4+2, rd:0; store.ugm.d16u32.a64
(W)     mul (1|M0)               acc0.0<1>:d   r4.1<0;1,0>:d     r3.4<0;1,0>:uw  
(W)     macl (1|M0)              r5.0<1>:d     r4.1<0;1,0>:d     r3.2<0;1,0>:d    {Compacted,$3.src}
        mov (16|M0)              r15.0<1>:bf   r62.0<1;1,0>:f                  
(W)     add (1|M0)               r2.4<1>:d     r2.7<0;1,0>:d     r5.0<0;1,0>:d    {I@1}
        mov (16|M16)             r15.16<1>:bf  r63.0<1;1,0>:f                  
        add (32|M0)              r6.0<1>:d     r22.0<1;1,0>:d    r2.4<0;1,0>:d    {Compacted,I@1}
        mov (16|M0)              r11.0<2>:ud   r6.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M16)             r13.0<2>:ud   r7.0<1;1,0>:ud                   {Compacted}
        shl (16|M0)              r9.0<1>:q     r11.0<2;1,0>:d    1:w               {I@2}
        shl (16|M16)             r11.0<1>:q    r13.0<2;1,0>:d    1:w               {I@2}
        add (16|M0)              r5.0<1>:q     r1.5<0;1,0>:q     r9.0<1;1,0>:q    {Compacted,I@2}
        add (16|M16)             r7.0<1>:q     r1.5<0;1,0>:q     r11.0<1;1,0>:q   {Compacted,I@2}
        mov (32|M0)              r10.0<1>:ud   r15.0<1;1,0>:uw                  {F@1}
        send.ugm (32|M0)         null     r5  r10:2  0x0            0x08000B84           {I@1,$4} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r6.0<1>:d     r52.0<1;1,0>:d    r2.4<0;1,0>:d    {Compacted,$4.src}
        mov (16|M0)              r11.0<2>:ud   r6.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M16)             r13.0<2>:ud   r7.0<1;1,0>:ud                   {Compacted}
        mov (16|M0)              r15.0<1>:bf   r60.0<1;1,0>:f                  
        mov (16|M16)             r15.16<1>:bf  r61.0<1;1,0>:f                  
        shl (16|M0)              r9.0<1>:q     r11.0<2;1,0>:d    1:w               {I@2}
        shl (16|M16)             r11.0<1>:q    r13.0<2;1,0>:d    1:w               {I@2}
        add (16|M0)              r5.0<1>:q     r1.5<0;1,0>:q     r9.0<1;1,0>:q    {Compacted,I@2}
        add (16|M16)             r7.0<1>:q     r1.5<0;1,0>:q     r11.0<1;1,0>:q   {Compacted,I@2}
        mov (32|M0)              r10.0<1>:ud   r15.0<1;1,0>:uw                  {F@1}
        send.ugm (32|M0)         null     r5  r10:2  0x0            0x08000B84           {I@1,$5} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r6.0<1>:d     r72.0<1;1,0>:d    r2.4<0;1,0>:d    {Compacted,$5.src}
        mov (16|M0)              r11.0<2>:ud   r6.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M16)             r13.0<2>:ud   r7.0<1;1,0>:ud                   {Compacted}
        mov (16|M0)              r15.0<1>:bf   r58.0<1;1,0>:f                  
        mov (16|M16)             r15.16<1>:bf  r59.0<1;1,0>:f                  
        shl (16|M0)              r9.0<1>:q     r11.0<2;1,0>:d    1:w               {I@2}
        shl (16|M16)             r11.0<1>:q    r13.0<2;1,0>:d    1:w               {I@2}
        add (16|M0)              r5.0<1>:q     r1.5<0;1,0>:q     r9.0<1;1,0>:q    {Compacted,I@2}
        add (16|M16)             r7.0<1>:q     r1.5<0;1,0>:q     r11.0<1;1,0>:q   {Compacted,I@2}
        mov (32|M0)              r10.0<1>:ud   r15.0<1;1,0>:uw                  {F@1}
        send.ugm (32|M0)         null     r5  r10:2  0x0            0x08000B84           {I@1,$6} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r6.0<1>:d     r74.0<1;1,0>:d    r2.4<0;1,0>:d    {Compacted,$6.src}
        mov (16|M0)              r11.0<2>:ud   r6.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M16)             r13.0<2>:ud   r7.0<1;1,0>:ud                   {Compacted}
        mov (16|M0)              r15.0<1>:bf   r56.0<1;1,0>:f                  
        mov (16|M16)             r15.16<1>:bf  r57.0<1;1,0>:f                  
        shl (16|M0)              r9.0<1>:q     r11.0<2;1,0>:d    1:w               {I@2}
        shl (16|M16)             r11.0<1>:q    r13.0<2;1,0>:d    1:w               {I@2}
(W)     or (1|M0)                r4.1<1>:d     r3.6<0;1,0>:d     2:w              
        add (16|M0)              r5.0<1>:q     r1.5<0;1,0>:q     r9.0<1;1,0>:q    {Compacted,I@3}
        add (16|M16)             r7.0<1>:q     r1.5<0;1,0>:q     r11.0<1;1,0>:q   {Compacted,I@3}
        mov (32|M0)              r10.0<1>:ud   r15.0<1;1,0>:uw                  {F@1}
        send.ugm (32|M0)         null     r5  r10:2  0x0            0x08000B84           {I@1,$7} // wr:4+2, rd:0; store.ugm.d16u32.a64
(W)     mul (1|M0)               acc0.0<1>:d   r4.1<0;1,0>:d     r3.4<0;1,0>:uw  
(W)     macl (1|M0)              r5.0<1>:d     r4.1<0;1,0>:d     r3.2<0;1,0>:d    {Compacted,$7.src}
        mov (16|M0)              r15.0<1>:bf   r54.0<1;1,0>:f                  
(W)     add (1|M0)               r2.4<1>:d     r2.7<0;1,0>:d     r5.0<0;1,0>:d    {I@1}
        mov (16|M16)             r15.16<1>:bf  r55.0<1;1,0>:f                  
        add (32|M0)              r6.0<1>:d     r22.0<1;1,0>:d    r2.4<0;1,0>:d    {Compacted,I@1}
        mov (16|M0)              r11.0<2>:ud   r6.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M16)             r13.0<2>:ud   r7.0<1;1,0>:ud                   {Compacted}
        shl (16|M0)              r9.0<1>:q     r11.0<2;1,0>:d    1:w               {I@2}
        shl (16|M16)             r11.0<1>:q    r13.0<2;1,0>:d    1:w               {I@2}
        add (16|M0)              r5.0<1>:q     r1.5<0;1,0>:q     r9.0<1;1,0>:q    {Compacted,I@2}
        add (16|M16)             r7.0<1>:q     r1.5<0;1,0>:q     r11.0<1;1,0>:q   {Compacted,I@2}
        mov (32|M0)              r10.0<1>:ud   r15.0<1;1,0>:uw                  {F@1}
        send.ugm (32|M0)         null     r5  r10:2  0x0            0x08000B84           {I@1,$8} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r6.0<1>:d     r52.0<1;1,0>:d    r2.4<0;1,0>:d    {Compacted,$8.src}
        mov (16|M0)              r11.0<2>:ud   r6.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M16)             r13.0<2>:ud   r7.0<1;1,0>:ud                   {Compacted}
        mov (16|M0)              r15.0<1>:bf   r50.0<1;1,0>:f                  
        mov (16|M16)             r15.16<1>:bf  r51.0<1;1,0>:f                  
        shl (16|M0)              r9.0<1>:q     r11.0<2;1,0>:d    1:w               {I@2}
        shl (16|M16)             r11.0<1>:q    r13.0<2;1,0>:d    1:w               {I@2}
        add (16|M0)              r5.0<1>:q     r1.5<0;1,0>:q     r9.0<1;1,0>:q    {Compacted,I@2}
        add (16|M16)             r7.0<1>:q     r1.5<0;1,0>:q     r11.0<1;1,0>:q   {Compacted,I@2}
        mov (32|M0)              r10.0<1>:ud   r15.0<1;1,0>:uw                  {F@1}
        send.ugm (32|M0)         null     r5  r10:2  0x0            0x08000B84           {I@1,$3} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r6.0<1>:d     r72.0<1;1,0>:d    r2.4<0;1,0>:d    {Compacted,$3.src}
        mov (16|M0)              r11.0<2>:ud   r6.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M16)             r13.0<2>:ud   r7.0<1;1,0>:ud                   {Compacted}
        mov (16|M0)              r15.0<1>:bf   r34.0<1;1,0>:f                  
        mov (16|M16)             r15.16<1>:bf  r35.0<1;1,0>:f                  
        shl (16|M0)              r9.0<1>:q     r11.0<2;1,0>:d    1:w               {I@2}
        shl (16|M16)             r11.0<1>:q    r13.0<2;1,0>:d    1:w               {I@2}
        add (16|M0)              r5.0<1>:q     r1.5<0;1,0>:q     r9.0<1;1,0>:q    {Compacted,I@2}
        add (16|M16)             r7.0<1>:q     r1.5<0;1,0>:q     r11.0<1;1,0>:q   {Compacted,I@2}
        mov (32|M0)              r10.0<1>:ud   r15.0<1;1,0>:uw                  {F@1}
        send.ugm (32|M0)         null     r5  r10:2  0x0            0x08000B84           {I@1,$4} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r6.0<1>:d     r74.0<1;1,0>:d    r2.4<0;1,0>:d    {Compacted,$4.src}
        mov (16|M0)              r11.0<2>:ud   r6.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M16)             r13.0<2>:ud   r7.0<1;1,0>:ud                   {Compacted}
        mov (16|M0)              r15.0<1>:bf   r32.0<1;1,0>:f                  
        mov (16|M16)             r15.16<1>:bf  r33.0<1;1,0>:f                  
        shl (16|M0)              r9.0<1>:q     r11.0<2;1,0>:d    1:w               {I@2}
        shl (16|M16)             r11.0<1>:q    r13.0<2;1,0>:d    1:w               {I@2}
(W)     or (1|M0)                r4.1<1>:d     r3.6<0;1,0>:d     3:w              
        add (16|M0)              r5.0<1>:q     r1.5<0;1,0>:q     r9.0<1;1,0>:q    {Compacted,I@3}
        add (16|M16)             r7.0<1>:q     r1.5<0;1,0>:q     r11.0<1;1,0>:q   {Compacted,I@3}
        mov (32|M0)              r10.0<1>:ud   r15.0<1;1,0>:uw                  {F@1}
        send.ugm (32|M0)         null     r5  r10:2  0x0            0x08000B84           {I@1,$5} // wr:4+2, rd:0; store.ugm.d16u32.a64
(W)     mul (1|M0)               acc0.0<1>:d   r4.1<0;1,0>:d     r3.4<0;1,0>:uw  
(W)     macl (1|M0)              r5.0<1>:d     r4.1<0;1,0>:d     r3.2<0;1,0>:d    {Compacted,$5.src}
        mov (16|M0)              r15.0<1>:bf   r30.0<1;1,0>:f                  
(W)     add (1|M0)               r2.4<1>:d     r2.7<0;1,0>:d     r5.0<0;1,0>:d    {I@1}
        mov (16|M16)             r15.16<1>:bf  r31.0<1;1,0>:f                  
        add (32|M0)              r6.0<1>:d     r22.0<1;1,0>:d    r2.4<0;1,0>:d    {Compacted,I@1}
        mov (16|M0)              r11.0<2>:ud   r6.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M16)             r13.0<2>:ud   r7.0<1;1,0>:ud                   {Compacted}
        shl (16|M0)              r9.0<1>:q     r11.0<2;1,0>:d    1:w               {I@2}
        shl (16|M16)             r11.0<1>:q    r13.0<2;1,0>:d    1:w               {I@2}
        add (16|M0)              r5.0<1>:q     r1.5<0;1,0>:q     r9.0<1;1,0>:q    {Compacted,I@2}
        add (16|M16)             r7.0<1>:q     r1.5<0;1,0>:q     r11.0<1;1,0>:q   {Compacted,I@2}
        mov (32|M0)              r10.0<1>:ud   r15.0<1;1,0>:uw                  {F@1}
        send.ugm (32|M0)         null     r5  r10:2  0x0            0x08000B84           {I@1,$6} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r6.0<1>:d     r52.0<1;1,0>:d    r2.4<0;1,0>:d    {Compacted,$6.src}
        mov (16|M0)              r11.0<2>:ud   r6.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M16)             r13.0<2>:ud   r7.0<1;1,0>:ud                   {Compacted}
        mov (16|M0)              r15.0<1>:bf   r28.0<1;1,0>:f                  
        mov (16|M16)             r15.16<1>:bf  r29.0<1;1,0>:f                  
        shl (16|M0)              r9.0<1>:q     r11.0<2;1,0>:d    1:w               {I@2}
        shl (16|M16)             r11.0<1>:q    r13.0<2;1,0>:d    1:w               {I@2}
        add (16|M0)              r5.0<1>:q     r1.5<0;1,0>:q     r9.0<1;1,0>:q    {Compacted,I@2}
        add (16|M16)             r7.0<1>:q     r1.5<0;1,0>:q     r11.0<1;1,0>:q   {Compacted,I@2}
        mov (32|M0)              r10.0<1>:ud   r15.0<1;1,0>:uw                  {F@1}
        send.ugm (32|M0)         null     r5  r10:2  0x0            0x08000B84           {I@1,$7} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r6.0<1>:d     r72.0<1;1,0>:d    r2.4<0;1,0>:d    {Compacted,$7.src}
        mov (16|M0)              r11.0<2>:ud   r6.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M16)             r13.0<2>:ud   r7.0<1;1,0>:ud                   {Compacted}
        mov (16|M0)              r15.0<1>:bf   r26.0<1;1,0>:f                  
        mov (16|M16)             r15.16<1>:bf  r27.0<1;1,0>:f                  
        shl (16|M0)              r9.0<1>:q     r11.0<2;1,0>:d    1:w               {I@2}
        shl (16|M16)             r11.0<1>:q    r13.0<2;1,0>:d    1:w               {I@2}
        add (16|M0)              r5.0<1>:q     r1.5<0;1,0>:q     r9.0<1;1,0>:q    {Compacted,I@2}
        add (16|M16)             r7.0<1>:q     r1.5<0;1,0>:q     r11.0<1;1,0>:q   {Compacted,I@2}
        mov (32|M0)              r10.0<1>:ud   r15.0<1;1,0>:uw                  {F@1}
        send.ugm (32|M0)         null     r5  r10:2  0x0            0x08000B84           {I@1,$8} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r6.0<1>:d     r74.0<1;1,0>:d    r2.4<0;1,0>:d    {Compacted,$8.src}
        mov (16|M0)              r11.0<2>:ud   r6.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M16)             r13.0<2>:ud   r7.0<1;1,0>:ud                   {Compacted}
        mov (16|M0)              r15.0<1>:bf   r24.0<1;1,0>:f                  
        mov (16|M16)             r15.16<1>:bf  r25.0<1;1,0>:f                  
        shl (16|M0)              r9.0<1>:q     r11.0<2;1,0>:d    1:w               {I@2}
        shl (16|M16)             r11.0<1>:q    r13.0<2;1,0>:d    1:w               {I@2}
        add (16|M0)              r5.0<1>:q     r1.5<0;1,0>:q     r9.0<1;1,0>:q    {Compacted,I@2}
        add (16|M16)             r7.0<1>:q     r1.5<0;1,0>:q     r11.0<1;1,0>:q   {Compacted,I@2}
        mov (32|M0)              r10.0<1>:ud   r15.0<1;1,0>:uw                  {F@1}
        send.ugm (32|M0)         null     r5  r10:2  0x0            0x08000B84           {I@1,$1} // wr:4+2, rd:0; store.ugm.d16u32.a64
L13968:
(W)     add (1|M0)               r4.1<1>:d     r1.15<0;1,0>:d    1:w              
(W)     cmp (32|M0)   (lt)f2.0   null<1>:d     r4.1<0;1,0>:d     r86.1<0;1,0>:d   {I@1}
(W&~f2.0) jmpi                               L14048                                
L14016:
(W)     mov (1|M0)               r1.15<1>:d    r4.1<0;1,0>:d                   
(W)     jmpi                                 L5080                                
L14048:
(W)     mov (16|M0)              r112.0<1>:f   r21.0<1;1,0>:f                   {Compacted}
(W)     send.gtwy (1|M0)         null     r112  null:0  0x0            0x02000010           {EOT,F@1,$3} // wr:1+0, rd:0; end of thread
L14072:
(W)     mov (16|M0)              null<1>:ud    0x82421A2:ud                             
(W)     mov (16|M0)              null<1>:ud    0xA1EF9BA8:ud                             
(W)     mov (16|M0)              null<1>:ud    0x0:ud                             
(W)     mov (16|M0)              null<1>:ud    0x1:ud                             
        illegal                
        illegal                
        illegal                
        illegal                
        illegal                
        illegal                
        illegal                
        illegal                
