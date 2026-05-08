L0:
(W)     and (1|M0)               r127.0<1>:ud  r0.0<0;1,0>:ud    0xFFFFFFC0:ud             
(W)     add (1|M0)               r127.0<1>:ud  r127.0<0;1,0>:ud  0x0:ud              {I@1}
(W)     send.ugm (1|M0)          r2       r127  null:0  0xFF000000            0x6219D500           {A@1,$0} // wr:1+0, rd:1; load.ugm.d32x16t.a32.ca.cc.bti[255]
(W)     send.ugm (1|M0)          r3       r127  null:0  0xFF040000            0x6219C500           {$1} // wr:1+0, rd:1; load.ugm.d32x8t.a32.ca.cc.bti[255][A+0x40]
(W)     mov (16|M0)              r55.0<1>:ud   r0.0<1;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               r54.0<1>:f    9.18355e-41:f                              
(W)     and (1|M0)               r1.9<1>:ud    r55.5<0;1,0>:ud   0xFFFFFC00:ud              {I@1}
(W)     or (1|M0)                cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x400004C0:ud              {A@1}
(W)     mov (8|M0)               r52.0<1>:w    0x76543210:v                               {A@1}
(W)     cmp (32|M0)   (eq)f3.0   null<1>:d     r3.1<0;1,0>:d     0:w               {$1.dst}
(W)     add (8|M0)               r52.8<1>:w    r52.0<1;1,0>:w    8:w               {I@2}
(W)     mov (1|M0)               r3.11<1>:d    r55.1<0;1,0>:d                  
(W)     add (16|M0)              r52.16<1>:w   r52.0<1;1,0>:w    16:w               {I@2}
(W)     mov (1|M0)               r3.48<2>:b    r55.8<0;1,0>:b                  
(W&~f3.0) jmpi                               L264                                
L232:
(W)     mov (1|M0)               r3.8<1>:d     -1:w                              
(W)     jmpi                                 L816                                
L264:
(W)     asr (1|M0)               r1.14<1>:d    r3.1<0;1,0>:d     31:w              
(W)     asr (1|M0)               r4.1<1>:d     r3.3<0;1,0>:d     31:w              
(W)     add (1|M0)               r1.10<1>:d    r1.14<0;1,0>:d    r3.1<0;1,0>:d    {I@2}
(W)     xor (1|M0)               r1.11<1>:d    r1.10<0;1,0>:d    r1.14<0;1,0>:d   {I@1}
(W)     add (1|M0)               r1.10<1>:d    r4.1<0;1,0>:d     r3.3<0;1,0>:d   
(W)     xor (1|M0)               r3.6<1>:d     r1.10<0;1,0>:d    r4.1<0;1,0>:d    {I@1}
(W)     xor (1|M0)               cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x30:uw              {A@1}
(W)     mov (1|M0)               r4.0<1>:f     r1.11<0;1,0>:ud                  {A@1}
(W)     mov (1|M0)               r1.15<1>:f    r3.6<0;1,0>:ud                   {I@2}
(W)     mov (1|M0)               r1.10<1>:ud   r4.0<0;1,0>:f                    {F@2}
(W)     math.inv (1|M0)          r4.2<1>:f     r4.0<0;1,0>:f                   
(W)     add (1|M0)               r1.12<1>:d    r1.11<0;1,0>:d    -r1.10<0;1,0>:d  {I@1}
(W)     mov (1|M0)               r1.10<1>:f    0xB4C00000:f                               {I@1}
(W)     mov (1|M0)               r54.4<1>:f    r1.12<0;1,0>:ud                 
(W)     mad (1|M0)               r3.10<1>:f    r4.2<0;0>:f       r1.10<0;0>:f      r4.2<0>:f        {A@1}
(W)     mov (1|M0)               r1.10<1>:ud   r1.15<0;1,0>:f                   {F@1}
(W)     mul (1|M0)               r3.7<1>:f     r1.15<0;1,0>:f    r3.10<0;1,0>:f  
(W)     add (1|M0)               r1.13<1>:d    r3.6<0;1,0>:d     -r1.10<0;1,0>:d  {I@1}
(W)     mov (1|M0)               r3.9<1>:ud    r3.7<0;1,0>:f                    {F@1}
(W)     mov (1|M0)               r54.5<1>:f    r1.13<0;1,0>:ud                  {I@2}
(W)     mov (1|M0)               r3.7<1>:f     r3.9<0;1,0>:ud                   {I@1}
(W)     mad (1|M0)               r1.12<1>:f    r1.15<0;0>:f      r3.7<0;0>:f       -r4.0<0>:f       {F@1}
(W)     mad (1|M0)               r1.10<1>:f    r54.5<0;0>:f      r3.7<0;0>:f       -r54.4<0>:f     
(W)     add (1|M0)               r1.10<1>:f    r1.12<0;1,0>:f    r1.10<0;1,0>:f   {F@1}
(W)     mul (1|M0)               r1.10<1>:f    r3.10<0;1,0>:f    r1.10<0;1,0>:f   {F@1}
(W)     xor (1|M0)               cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x30:uw              {A@1}
(W)     mov (1|M0)               r1.10<1>:ud   r1.10<0;1,0>:f                   {A@1}
(W)     xor (1|M0)               r1.13<1>:d    r1.14<0;1,0>:d    r4.1<0;1,0>:d   
(W)     add (1|M0)               r1.12<1>:d    r1.10<0;1,0>:d    r3.9<0;1,0>:d    {I@2}
(W)     mul (1|M0)               acc0.0<1>:d   r1.12<0;1,0>:d    r1.22<0;1,0>:uw  {I@1}
(W)     macl (1|M0)              r4.0<1>:d     r1.12<0;1,0>:d    r1.11<0;1,0>:d   {Compacted}
(W)     add (1|M0)               r1.10<1>:d    r3.6<0;1,0>:d     -r4.0<0;1,0>:d   {I@1}
(W)     cmp (1|M0)    (ge)f0.0   r54.1<1>:ud   r1.10<0;1,0>:ud   r1.11<0;1,0>:ud  {I@1}
(W)     add3 (1|M0)              r1.10<1>:d    r1.12<0;0>:d      r1.13<0;0>:d      -r54.1<0>:d      {I@1}
(W)     bfn.(s0^s1^s2) (1|M0)    r3.8<1>:ud    r1.10<0;0>:ud     r1.14<0;0>:ud     r4.1<0>:ud       {I@1}
L816:
(W)     mov (1|M0)               r1.10<1>:d    r3.48<0;1,0>:ub                 
(W)     shl (1|M0)               r1.11<1>:d    r3.11<0;1,0>:d    5:w              
(W)     shl (1|M0)               r1.10<1>:d    r1.10<0;1,0>:d    2:w               {I@2}
(W)     add (1|M0)               r3.6<1>:d     r1.11<0;1,0>:d    r1.10<0;1,0>:d   {I@1}
(W)     cmp (32|M0)   (lt)f2.0   null<1>:d     r3.6<0;1,0>:d     r3.4<0;1,0>:d    {I@1}
(W&~f2.0) jmpi                               L13248                                
L912:
(W)     mov (1|M0)               r3.7<1>:d     r55.6<0;1,0>:d                  
(W)     shl (1|M0)               r1.6<1>:q     r55.7<0;1,0>:ud   2:w              
(W)     shl (1|M0)               r1.5<1>:q     r3.7<0;1,0>:ud    2:w               {I@2}
(W)     shl (1|M0)               r1.7<1>:q     r3.7<0;1,0>:ud    1:w              
(W)     add (1|M0)               r10.0<1>:q    r1.5<0;1,0>:q     r2.2<0;1,0>:q    {Compacted,@2,$0.dst}
(W)     add (1|M0)               r6.0<1>:q     r1.6<0;1,0>:q     r2.7<0;1,0>:q    {Compacted}
(W)     send.ugm (1|M0)          r7       r10  null:0  0x0            0x02108580           {I@2,$2} // wr:1+0, rd:1; load.ugm.d32x1t.a64
(W)     send.ugm (1|M0)          r8       r6  null:0  0x0            0x02108580           {I@1,$3} // wr:1+0, rd:1; load.ugm.d32x1t.a64
(W)     add (1|M0)               r6.0<1>:q     r1.7<0;1,0>:q     r2.3<0;1,0>:q    {Compacted,$3.src}
(W)     mul (1|M0)               acc0.0<1>:d   r3.7<0;1,0>:d     r3.4<0;1,0>:uw  
(W)     send.ugm (1|M0)          r5       r6  null:0  0x0            0x04100B80           {I@2,$2} // wr:2+0, rd:1; load.ugm.d16u32.a64
(W)     macl (1|M0)              r4.0<1>:d     r3.7<0;1,0>:d     r3.2<0;1,0>:d   
(W)     mov (1|M0)               r1.8<1>:ud    a0.2<0;1,0>:ud                  
(W)     mul (1|M0)               acc0.0<1>:d   r4.0<0;1,0>:d     r3.8<0;1,0>:uw   {I@2}
(W)     macl (1|M0)              r4.0<1>:d     r4.0<0;1,0>:d     r3.4<0;1,0>:d    {Compacted}
(W)     mul (1|M0)               acc0.0<1>:d   r55.7<0;1,0>:d    r3.10<0;1,0>:uw 
(W)     shr (1|M0)               a0.2<1>:ud    r1.9<0;1,0>:ud    0x4:ud              {F@1}
(W)     mov (1|M0)               r2.6<1>:d     r4.0<0;1,0>:d                    {I@3}
(W)     macl (1|M0)              r4.0<1>:d     r55.7<0;1,0>:d    r3.5<0;1,0>:d   
        shl (32|M0)              r16.0<1>:d    r52.0<1;1,0>:uw   2:w              
(W)     mov (1|M0)               r2.14<1>:d    r4.0<0;1,0>:d                    {I@2}
(W)     mov (1|M0)               r3.10<1>:f    0xBF317200:f                              
(W)     mov (1|M0)               r3.9<1>:f     0xB5BFBE8E:f                              
(W)     mul (1|M0)               r1.10<1>:f    r7.0<0;1,0>:f     1.442695e+00:f              
(W)     cmp (1|M0)    (lt)f3.0   null<1>:f     r7.0<0;1,0>:f     -105.0:f              
(W)     rndz (1|M0)              r6.0<1>:f     r1.10<0;1,0>:f                   {Compacted,@2,$2.src}
(W)     mov (1|M0)               r1.10<1>:f    0xBF317200:f                              
(W)     cmp (1|M0)    (gt)f2.0   null<1>:f     r7.0<0;1,0>:f     105.0:f              
(W)     mad (1|M0)               r1.11<1>:f    r7.0<0;0>:f       r1.10<0;0>:f      r6.0<0>:f        {F@2}
(W)     mov (1|M0)               r1.10<1>:f    0xB5BFBE8E:f                              
(W)     math.exp (1|M0)          r7.0<1>:f     r6.0<0;1,0>:f                    {F@2}
(W)     mad (1|M0)               r1.10<1>:f    r1.11<0;0>:f      r1.10<0;0>:f      r6.0<0>:f        {F@1}
(W)     cmp (32|M0)   (lt)f1.0   null<1>:d     r8.0<0;1,0>:d     1:w               {$3.dst}
(W)     mul (1|M0)               r6.1<1>:f     r1.10<0;1,0>:f    1.442695e+00:f               {F@1}
(W)     shl (1|M0)               r4.0<1>:ud    r5.0<0;1,0>:uw    0x10:uw              {$2.dst}
(W)     math.exp (1|M0)          r7.1<1>:f     r6.1<0;1,0>:f                    {F@1}
(W)     mul (1|M0)               r1.10<1>:f    r7.0<0;1,0>:f     r7.1<0;1,0>:f    {M@1}
(W&~f3.0) sel (1|M0)             r1.10<1>:f    r1.10<0;1,0>:f    0.0:f               {F@1}
(W&~f2.0) sel (1|M0)             r5.0<1>:f     r1.10<0;1,0>:f    inf:f               {A@1}
(W)     send.ugm (1|M0)          null     r54  r4:2  0x80000000:a0.2        0x4200E504           {F@1,$4} // wr:1+2, rd:0; store.ugm.d32x32t.a32.ss[a0.2][A-0x10000]
(W)     mov (1|M0)               a0.2<1>:ud    r1.8<0;1,0>:ud                   {$4.src}
(W&f1.0) jmpi                                L1808                                
L1544:
(W)     add3 (1|M0)              r1.10<1>:d    r2.14<0;0>:d      r8.0<0;0>:d       -1:w              
(W)     shl (1|M0)               r1.5<1>:q     r1.10<0;1,0>:d    2:w               {I@1}
(W)     add (1|M0)               r4.0<1>:q     r1.5<0;1,0>:q     r2.6<0;1,0>:q    {I@1}
(W)     asr (1|M0)               r1.10<1>:d    r3.0<0;1,0>:d     31:w              
(W)     send.ugm (1|M0)          r4       r4  null:0  0x0            0x02108580           {I@2,$5} // wr:1+0, rd:1; load.ugm.d32x1t.a64
(W)     mul (1|M0)               acc0.0<1>:ud  r4.0<0;1,0>:ud    r3.0<0;1,0>:uw   {$5.dst}
(W)     macl (1|M0)              r7.0<1>:ud    r4.0<0;1,0>:ud    r3.0<0;1,0>:ud   {Compacted}
(W)     mul (1|M0)               acc0.0<1>:ud  r4.0<0;1,0>:ud    r3.0<0;1,0>:uw  
(W)     asr (1|M0)               r1.11<1>:d    r4.0<0;1,0>:d     31:w              
(W)     mach (1|M0)              r6.0<1>:d     r4.0<0;1,0>:ud    r3.0<0;1,0>:ud  
(W)     mul (1|M0)               acc0.0<1>:d   r4.0<0;1,0>:ud    r1.20<0;1,0>:uw  {I@6}
(W)     cmp (32|M0)   (gt)f1.0   null<1>:d     r4.0<0;1,0>:d     0:w              
(W)     macl (1|M0)              r5.0<1>:d     r4.0<0;1,0>:ud    r1.10<0;1,0>:d  
(W)     mul (1|M0)               acc0.0<1>:d   r3.0<0;1,0>:ud    r1.22<0;1,0>:uw  {I@5}
(W)     add (1|M0)               r6.0<1>:d     r6.0<0;1,0>:d     r5.0<0;1,0>:d    {Compacted,I@2}
(W)     macl (1|M0)              r5.0<1>:d     r3.0<0;1,0>:ud    r1.11<0;1,0>:d  
(W)     add (1|M0)               r2.4<1>:d     r6.0<0;1,0>:d     r5.0<0;1,0>:d    {Compacted,I@1}
(W&f1.0) jmpi                                L1952                                
L1808:
        mov (32|M0)              r34.0<1>:ud   0x0:ud                              {Compacted}
        mov (32|M0)              r56.0<1>:ud   0x0:ud                              {Compacted}
        mov (32|M0)              r58.0<1>:ud   0x0:ud                              {Compacted}
        mov (32|M0)              r60.0<1>:ud   0x0:ud                              {Compacted}
        mov (32|M0)              r62.0<1>:ud   0x0:ud                              {Compacted}
        mov (32|M0)              r64.0<1>:ud   0x0:ud                              {Compacted}
        mov (32|M0)              r66.0<1>:ud   0x0:ud                              {Compacted}
        mov (32|M0)              r68.0<1>:ud   0x0:ud                              {Compacted}
        mov (32|M0)              r32.0<1>:f    0.0:f                               {Compacted}
        mov (32|M0)              r30.0<1>:f    0.0:f                               {Compacted}
        mov (32|M0)              r28.0<1>:f    0.0:f                               {Compacted}
        mov (32|M0)              r26.0<1>:f    0.0:f                               {Compacted}
        mov (32|M0)              r24.0<1>:f    0.0:f                               {Compacted}
        mov (32|M0)              r22.0<1>:f    0.0:f                               {Compacted}
        mov (32|M0)              r20.0<1>:f    0.0:f                               {Compacted}
        mov (32|M0)              r18.0<1>:f    0.0:f                               {Compacted}
(W)     jmpi                                 L4512                                
L1952:
(W)     or (1|M0)                r1.14<1>:d    r3.6<0;1,0>:d     1:w              
(W)     mul (1|M0)               acc0.0<1>:d   r3.6<0;1,0>:d     r3.4<0;1,0>:uw  
(W)     or (1|M0)                r1.15<1>:d    r3.6<0;1,0>:d     2:w              
(W)     or (1|M0)                r3.5<1>:d     r3.6<0;1,0>:d     3:w              
(W)     macl (1|M0)              r4.0<1>:d     r3.6<0;1,0>:d     r3.2<0;1,0>:d    {Compacted}
(W)     mul (1|M0)               acc0.0<1>:d   r1.14<0;1,0>:d    r3.4<0;1,0>:uw   {I@5}
(W)     macl (1|M0)              r6.0<1>:d     r1.14<0;1,0>:d    r3.2<0;1,0>:d    {Compacted}
(W)     mul (1|M0)               acc0.0<1>:d   r1.15<0;1,0>:d    r3.4<0;1,0>:uw   {I@5}
(W)     macl (1|M0)              r5.0<1>:d     r1.15<0;1,0>:d    r3.2<0;1,0>:d   
(W)     mul (1|M0)               acc0.0<1>:d   r3.5<0;1,0>:d     r3.4<0;1,0>:uw   {I@6}
(W)     add (1|M0)               r2.7<1>:d     r2.6<0;1,0>:d     r4.0<0;1,0>:d    {I@6}
(W)     macl (1|M0)              r4.0<1>:d     r3.5<0;1,0>:d     r3.2<0;1,0>:d   
(W)     mov (1|M0)               r1.11<1>:d    r2.4<0;1,0>:d                   
(W)     add (1|M0)               r2.5<1>:d     r2.6<0;1,0>:d     r5.0<0;1,0>:d    {I@5}
(W)     add (1|M0)               r2.4<1>:d     r2.6<0;1,0>:d     r4.0<0;1,0>:d    {I@3}
        add (32|M0)              r4.0<1>:d     r16.0<1;1,0>:d    r2.7<0;1,0>:d    {Compacted}
        mov (16|M0)              r48.0<2>:ud   r4.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M16)             r46.0<2>:ud   r5.0<1;1,0>:ud                   {Compacted}
        add3 (32|M0)             r4.0<1>:d     r16.0<1;0>:d      r2.7<0;0>:d       1:w              
        mov (16|M0)              r44.0<2>:ud   r4.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M16)             r42.0<2>:ud   r5.0<1;1,0>:ud                   {Compacted}
        add3 (32|M0)             r4.0<1>:d     r16.0<1;0>:d      r2.7<0;0>:d       2:w              
        mov (16|M0)              r40.0<2>:ud   r4.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M16)             r38.0<2>:ud   r5.0<1;1,0>:ud                   {Compacted}
        add3 (32|M0)             r4.0<1>:d     r16.0<1;0>:d      r2.7<0;0>:d       3:w              
(W)     add (1|M0)               r2.15<1>:d    r2.6<0;1,0>:d     r6.0<0;1,0>:d   
        mov (16|M0)              r36.0<2>:ud   r4.0<1;1,0>:ud                   {Compacted,I@2}
        mov (16|M16)             r14.0<2>:ud   r5.0<1;1,0>:ud                   {Compacted}
        add (32|M0)              r4.0<1>:d     r16.0<1;1,0>:d    r2.15<0;1,0>:d   {Compacted,I@3}
        mov (16|M0)              r12.0<2>:ud   r4.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M16)             r70.0<2>:ud   r5.0<1;1,0>:ud                   {Compacted}
        add3 (32|M0)             r4.0<1>:d     r16.0<1;0>:d      r2.15<0;0>:d      1:w              
        mov (16|M0)              r50.0<2>:ud   r4.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M16)             r74.0<2>:ud   r5.0<1;1,0>:ud                   {Compacted}
        add3 (32|M0)             r4.0<1>:d     r16.0<1;0>:d      r2.15<0;0>:d      2:w              
        mov (16|M0)              r72.0<2>:ud   r4.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M16)             r78.0<2>:ud   r5.0<1;1,0>:ud                   {Compacted}
        add3 (32|M0)             r4.0<1>:d     r16.0<1;0>:d      r2.15<0;0>:d      3:w              
        mov (16|M0)              r76.0<2>:ud   r4.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M16)             r82.0<2>:ud   r5.0<1;1,0>:ud                   {Compacted}
        add (32|M0)              r4.0<1>:d     r16.0<1;1,0>:d    r2.5<0;1,0>:d    {Compacted}
        mov (16|M0)              r80.0<2>:ud   r4.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M16)             r86.0<2>:ud   r5.0<1;1,0>:ud                   {Compacted}
        add3 (32|M0)             r4.0<1>:d     r16.0<1;0>:d      r2.5<0;0>:d       1:w              
        mov (16|M0)              r84.0<2>:ud   r4.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M16)             r90.0<2>:ud   r5.0<1;1,0>:ud                   {Compacted}
        add3 (32|M0)             r4.0<1>:d     r16.0<1;0>:d      r2.5<0;0>:d       2:w              
        mov (16|M0)              r88.0<2>:ud   r4.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M16)             r94.0<2>:ud   r5.0<1;1,0>:ud                   {Compacted}
        add3 (32|M0)             r4.0<1>:d     r16.0<1;0>:d      r2.5<0;0>:d       3:w              
        mov (16|M0)              r92.0<2>:ud   r4.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M16)             r98.0<2>:ud   r5.0<1;1,0>:ud                   {Compacted}
        add (32|M0)              r4.0<1>:d     r16.0<1;1,0>:d    r2.4<0;1,0>:d    {Compacted}
        mov (16|M0)              r96.0<2>:ud   r4.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M16)             r102.0<2>:ud  r5.0<1;1,0>:ud                   {Compacted}
        add3 (32|M0)             r4.0<1>:d     r16.0<1;0>:d      r2.4<0;0>:d       1:w              
        add3 (32|M0)             r8.0<1>:d     r16.0<1;0>:d      r2.4<0;0>:d       3:w              
        mov (16|M0)              r100.0<2>:ud  r4.0<1;1,0>:ud                   {Compacted,I@2}
        mov (16|M16)             r106.0<2>:ud  r5.0<1;1,0>:ud                   {Compacted}
        add3 (32|M0)             r4.0<1>:d     r16.0<1;0>:d      r2.4<0;0>:d       2:w              
(W)     mov (1|M0)               r1.8<1>:ud    a0.2<0;1,0>:ud                  
(W)     shr (1|M0)               a0.2<1>:ud    r1.9<0;1,0>:ud    0x4:ud             
        mov (16|M0)              r104.0<2>:ud  r4.0<1;1,0>:ud                   {Compacted,I@3}
        mov (16|M16)             r112.0<2>:ud  r5.0<1;1,0>:ud                   {Compacted}
        mov (16|M0)              r4.0<2>:ud    r8.0<1;1,0>:ud                   {Compacted}
(W)     send.ugm (1|M0)          null     r54  r4:2  0x82800000:a0.2        0x4200E504           {I@1,$6} // wr:1+2, rd:0; store.ugm.d32x32t.a32.ss[a0.2][A-0xFB00]
(W)     mov (1|M0)               r1.10<1>:d    r7.0<0;1,0>:d                   
        mov (16|M16)             r4.0<2>:ud    r9.0<1;1,0>:ud                   {Compacted,$6.src}
        shl (16|M16)             r6.0<1>:q     r46.0<2;1,0>:d    1:w              
(W)     send.ugm (1|M0)          null     r54  r4:4  0x82000000:a0.2        0x4200F504           {I@1,$7} // wr:1+4, rd:0; store.ugm.d32x64t.a32.ss[a0.2][A-0xFC00]
        shl (16|M16)             r4.0<1>:q     r42.0<2;1,0>:d    1:w               {$7.src}
        shl (16|M0)              r6.0<1>:q     r40.0<2;1,0>:d    1:w              
(W)     send.ugm (1|M0)          null     r54  r4:4  0x80C00000:a0.2        0x4200F504           {I@1,$8} // wr:1+4, rd:0; store.ugm.d32x64t.a32.ss[a0.2][A-0xFE80]
        shl (16|M16)             r4.0<1>:q     r38.0<2;1,0>:d    1:w               {$8.src}
        shl (16|M0)              r6.0<1>:q     r36.0<2;1,0>:d    1:w              
(W)     send.ugm (1|M0)          null     r54  r4:4  0x81400000:a0.2        0x4200F504           {I@1,$9} // wr:1+4, rd:0; store.ugm.d32x64t.a32.ss[a0.2][A-0xFD80]
        shl (16|M0)              r4.0<1>:q     r12.0<2;1,0>:d    1:w               {$9.src}
(W)     send.ugm (1|M0)          null     r54  r4:2  0x81C00000:a0.2        0x4200E504           {I@1,$10} // wr:1+2, rd:0; store.ugm.d32x32t.a32.ss[a0.2][A-0xFC80]
        shl (16|M16)             r108.0<1>:q   r112.0<2;1,0>:d   1:w              
(W)     shl (1|M0)               r1.5<1>:q     r1.5<0;1,0>:q     1:w              
        shl (16|M16)             r110.0<1>:q   r106.0<2;1,0>:d   1:w              
(W)     add (1|M0)               r1.5<1>:q     r1.5<0;1,0>:q     r2.4<0;1,0>:q    {I@2}
        shl (16|M0)              r114.0<1>:q   r104.0<2;1,0>:d   1:w              
        shl (16|M0)              r10.0<1>:q    r44.0<2;1,0>:d    1:w              
        shl (16|M0)              r8.0<1>:q     r48.0<2;1,0>:d    1:w              
(W)     send.ugm (1|M0)          null     r54  r8:4  0x80400000:a0.2        0x4200F504           {I@1,$11} // wr:1+4, rd:0; store.ugm.d32x64t.a32.ss[a0.2][A-0xFF80]
        shl (16|M0)              r116.0<1>:q   r100.0<2;1,0>:d   1:w              
        shl (16|M16)             r36.0<1>:q    r94.0<2;1,0>:d    1:w              
        shl (16|M0)              r118.0<1>:q   r96.0<2;1,0>:d    1:w              
        shl (16|M16)             r124.0<1>:q   r14.0<2;1,0>:d    1:w              
        shl (16|M16)             r122.0<1>:q   r70.0<2;1,0>:d    1:w              
        shl (16|M0)              r126.0<1>:q   r50.0<2;1,0>:d    1:w              
        shl (16|M0)              r42.0<1>:q    r88.0<2;1,0>:d    1:w              
        shl (16|M16)             r38.0<1>:q    r90.0<2;1,0>:d    1:w              
        shl (16|M16)             r12.0<1>:q    r102.0<2;1,0>:d   1:w              
        sync.nop                             null                             {Compacted,$10.src}
(W)     send.ugm (1|M0)          r4       r54  null:0  0x82800000:a0.2        0x4220E500           {$12} // wr:1+0, rd:2; load.ugm.d32x32t.a32.ss[a0.2][A-0xFB00]
        shl (16|M16)             r44.0<1>:q    r82.0<2;1,0>:d    1:w              
        shl (16|M16)             r14.0<1>:q    r98.0<2;1,0>:d    1:w              
        shl (16|M16)             r70.0<1>:q    r74.0<2;1,0>:d    1:w              
        shl (16|M0)              r50.0<1>:q    r80.0<2;1,0>:d    1:w              
        shl (16|M0)              r120.0<1>:q   r92.0<2;1,0>:d    1:w              
        shl (16|M0)              r46.0<1>:q    r84.0<2;1,0>:d    1:w              
        shl (16|M16)             r40.0<1>:q    r86.0<2;1,0>:d    1:w              
        shl (16|M16)             r48.0<1>:q    r78.0<2;1,0>:d    1:w              
        shl (16|M0)              r74.0<1>:q    r72.0<2;1,0>:d    1:w              
        add (16|M16)             r84.0<1>:q    r1.5<0;1,0>:q     r70.0<1;1,0>:q   {Compacted,I@7}
        shl (16|M0)              r72.0<1>:q    r76.0<2;1,0>:d    1:w              
        add (16|M0)              r70.0<1>:q    r1.5<0;1,0>:q     r50.0<1;1,0>:q   {Compacted,I@7}
        add (16|M16)             r76.0<1>:q    r1.5<0;1,0>:q     r44.0<1;1,0>:q   {Compacted}
        add (16|M16)             r50.0<1>:q    r1.5<0;1,0>:q     r38.0<1;1,0>:q   {Compacted}
        add (16|M0)              r44.0<1>:q    r1.5<0;1,0>:q     r42.0<1;1,0>:q   {Compacted}
        sync.nop                             null                             {Compacted,$11.src}
(W)     send.ugm (1|M0)          r8       r54  null:0  0x80400000:a0.2        0x4240F500           {$13} // wr:1+0, rd:4; load.ugm.d32x64t.a32.ss[a0.2][A-0xFF80]
        add (16|M16)             r38.0<1>:q    r1.5<0;1,0>:q     r12.0<1;1,0>:q   {Compacted}
        add (16|M16)             r42.0<1>:q    r1.5<0;1,0>:q     r14.0<1;1,0>:q   {Compacted}
        add (16|M0)              r12.0<1>:q    r1.5<0;1,0>:q     r116.0<1;1,0>:q  {Compacted}
        add (16|M16)             r14.0<1>:q    r1.5<0;1,0>:q     r110.0<1;1,0>:q  {Compacted}
        send.ugm (32|M0)         r18      r12  null:0  0x0            0x08200B80           {A@1,$14} // wr:4+0, rd:2; load.ugm.d16u32.a64
        add (16|M16)             r88.0<1>:q    r1.5<0;1,0>:q     r122.0<1;1,0>:q  {Compacted}
        add (16|M0)              r82.0<1>:q    r1.5<0;1,0>:q     r126.0<1;1,0>:q  {Compacted}
        add (16|M16)             r92.0<1>:q    r1.5<0;1,0>:q     r124.0<1;1,0>:q  {Compacted}
        shl (16|M0)              r112.0<1>:q   r4.0<2;1,0>:d     1:w               {$12.dst}
(W)     send.ugm (1|M0)          r4       r54  null:0  0x82000000:a0.2        0x4240F500           {I@1,$15} // wr:1+0, rd:4; load.ugm.d32x64t.a32.ss[a0.2][A-0xFC00]
        add (16|M16)             r80.0<1>:q    r1.5<0;1,0>:q     r48.0<1;1,0>:q   {Compacted}
        add (16|M0)              r78.0<1>:q    r1.5<0;1,0>:q     r74.0<1;1,0>:q   {Compacted}
        add (16|M0)              r48.0<1>:q    r1.5<0;1,0>:q     r46.0<1;1,0>:q   {Compacted}
        add (16|M0)              r74.0<1>:q    r1.5<0;1,0>:q     r72.0<1;1,0>:q   {Compacted}
        add (16|M16)             r46.0<1>:q    r1.5<0;1,0>:q     r36.0<1;1,0>:q   {Compacted}
        add (16|M16)             r72.0<1>:q    r1.5<0;1,0>:q     r40.0<1;1,0>:q   {Compacted}
        add (16|M0)              r36.0<1>:q    r1.5<0;1,0>:q     r118.0<1;1,0>:q  {Compacted}
        add (16|M0)              r40.0<1>:q    r1.5<0;1,0>:q     r120.0<1;1,0>:q  {Compacted}
        send.ugm (32|M0)         r34      r82  null:0  0x0            0x08200B80           {$0} // wr:4+0, rd:2; load.ugm.d16u32.a64
        send.ugm (32|M0)         r32      r78  null:0  0x0            0x08200B80           {I@7,$1} // wr:4+0, rd:2; load.ugm.d16u32.a64
        send.ugm (32|M0)         r26      r48  null:0  0x0            0x08200B80           {I@6,$2} // wr:4+0, rd:2; load.ugm.d16u32.a64
        add (16|M0)              r102.0<1>:q   r1.5<0;1,0>:q     r8.0<1;1,0>:q    {Compacted,$13.dst}
        add (16|M0)              r98.0<1>:q    r1.5<0;1,0>:q     r10.0<1;1,0>:q   {Compacted}
        add (16|M0)              r8.0<1>:q     r1.5<0;1,0>:q     r114.0<1;1,0>:q  {Compacted}
        add (16|M16)             r10.0<1>:q    r1.5<0;1,0>:q     r108.0<1;1,0>:q  {Compacted}
        send.ugm (32|M0)         r12      r8  null:0  0x0            0x08200B80           {I@1,$3} // wr:4+0, rd:2; load.ugm.d16u32.a64
        send.ugm (32|M0)         r30      r74  null:0  0x0            0x08200B80           {$4} // wr:4+0, rd:2; load.ugm.d16u32.a64
        send.ugm (32|M0)         r24      r44  null:0  0x0            0x08200B80           {$5} // wr:4+0, rd:2; load.ugm.d16u32.a64
        send.ugm (32|M0)         r28      r70  null:0  0x0            0x08200B80           {$6} // wr:4+0, rd:2; load.ugm.d16u32.a64
        shl (16|M16)             r106.0<1>:q   r4.0<2;1,0>:d     1:w               {$15.dst}
        add (16|M16)             r104.0<1>:q   r1.5<0;1,0>:q     r6.0<1;1,0>:q    {Compacted}
(W)     send.ugm (1|M0)          r4       r54  null:0  0x80C00000:a0.2        0x4240F500           {I@1,$7} // wr:1+0, rd:4; load.ugm.d32x64t.a32.ss[a0.2][A-0xFE80]
        send.ugm (32|M0)         r20      r36  null:0  0x0            0x08200B80           {$8} // wr:4+0, rd:2; load.ugm.d16u32.a64
        send.ugm (32|M0)         r22      r40  null:0  0x0            0x08200B80           {$9} // wr:4+0, rd:2; load.ugm.d16u32.a64
        send.ugm (32|M0)         r64      r102  null:0  0x0            0x08200B80           {$10} // wr:4+0, rd:2; load.ugm.d16u32.a64
        add (16|M16)             r100.0<1>:q   r1.5<0;1,0>:q     r4.0<1;1,0>:q    {Compacted,$7.dst}
        add (16|M0)              r94.0<1>:q    r1.5<0;1,0>:q     r6.0<1;1,0>:q    {Compacted}
(W)     send.ugm (1|M0)          r4       r54  null:0  0x81400000:a0.2        0x4240F500           {I@1,$11} // wr:1+0, rd:4; load.ugm.d32x64t.a32.ss[a0.2][A-0xFD80]
        send.ugm (32|M0)         r62      r98  null:0  0x0            0x08200B80           {$12} // wr:4+0, rd:2; load.ugm.d16u32.a64
        shl (16|M0)              r68.0<1>:ud   r64.0<2;1,0>:uw   0x10:uw              {$10.dst}
        shl (16|M16)             r69.0<1>:ud   r65.0<2;1,0>:uw   0x10:uw             
        add (16|M16)             r96.0<1>:q    r1.5<0;1,0>:q     r4.0<1;1,0>:q    {Compacted,$11.dst}
(W)     send.ugm (1|M0)          r4       r54  null:0  0x81C00000:a0.2        0x4220E500           {I@1,$13} // wr:1+0, rd:2; load.ugm.d32x32t.a32.ss[a0.2][A-0xFC80]
        add (16|M0)              r90.0<1>:q    r1.5<0;1,0>:q     r6.0<1;1,0>:q    {Compacted}
        add (16|M16)             r6.0<1>:q     r1.5<0;1,0>:q     r106.0<1;1,0>:q  {Compacted}
        send.ugm (32|M0)         r60      r94  null:0  0x0            0x08200B80           {$15} // wr:4+0, rd:2; load.ugm.d16u32.a64
        send.ugm (32|M0)         r58      r90  null:0  0x0            0x08200B80           {I@2,$7} // wr:4+0, rd:2; load.ugm.d16u32.a64
        shl (16|M0)              r66.0<1>:ud   r62.0<2;1,0>:uw   0x10:uw              {$12.dst}
        shl (16|M16)             r67.0<1>:ud   r63.0<2;1,0>:uw   0x10:uw             
(W)     mov (1|M0)               a0.2<1>:ud    r1.8<0;1,0>:ud                   {$13.src}
        add (16|M0)              r86.0<1>:q    r1.5<0;1,0>:q     r4.0<1;1,0>:q    {Compacted,$13.dst}
        add (16|M0)              r4.0<1>:q     r1.5<0;1,0>:q     r112.0<1;1,0>:q  {Compacted}
        send.ugm (32|M0)         r56      r86  null:0  0x0            0x08200B80           {I@2,$10} // wr:4+0, rd:2; load.ugm.d16u32.a64
        send.ugm (32|M0)         r8       r4  null:0  0x0            0x08200B80           {I@1,$11} // wr:4+0, rd:2; load.ugm.d16u32.a64
        shl (16|M0)              r64.0<1>:ud   r60.0<2;1,0>:uw   0x10:uw              {$15.dst}
        shl (16|M16)             r65.0<1>:ud   r61.0<2;1,0>:uw   0x10:uw             
        shl (16|M0)              r62.0<1>:ud   r58.0<2;1,0>:uw   0x10:uw              {$7.dst}
        shl (16|M16)             r63.0<1>:ud   r59.0<2;1,0>:uw   0x10:uw             
        shl (16|M0)              r58.0<1>:ud   r34.0<2;1,0>:uw   0x10:uw              {$0.dst}
        shl (16|M16)             r59.0<1>:ud   r35.0<2;1,0>:uw   0x10:uw             
        shl (16|M0)              r34.0<1>:ud   r30.0<2;1,0>:uw   0x10:uw              {$4.dst}
        shl (16|M16)             r35.0<1>:ud   r31.0<2;1,0>:uw   0x10:uw             
        shl (16|M0)              r30.0<1>:ud   r26.0<2;1,0>:uw   0x10:uw              {$2.dst}
        shl (16|M16)             r31.0<1>:ud   r27.0<2;1,0>:uw   0x10:uw             
        shl (16|M0)              r26.0<1>:ud   r22.0<2;1,0>:uw   0x10:uw              {$9.dst}
        shl (16|M16)             r27.0<1>:ud   r23.0<2;1,0>:uw   0x10:uw             
        shl (16|M0)              r22.0<1>:ud   r18.0<2;1,0>:uw   0x10:uw              {$14.dst}
        shl (16|M16)             r23.0<1>:ud   r19.0<2;1,0>:uw   0x10:uw             
        shl (16|M0)              r60.0<1>:ud   r56.0<2;1,0>:uw   0x10:uw              {$10.dst}
        shl (16|M16)             r61.0<1>:ud   r57.0<2;1,0>:uw   0x10:uw             
        shl (16|M0)              r56.0<1>:ud   r32.0<2;1,0>:uw   0x10:uw              {$1.dst}
        shl (16|M16)             r57.0<1>:ud   r33.0<2;1,0>:uw   0x10:uw             
        shl (16|M0)              r18.0<1>:ud   r8.0<2;1,0>:uw    0x10:uw              {$11.dst}
        shl (16|M16)             r19.0<1>:ud   r9.0<2;1,0>:uw    0x10:uw             
        shl (16|M0)              r32.0<1>:ud   r28.0<2;1,0>:uw   0x10:uw              {$6.dst}
        shl (16|M16)             r33.0<1>:ud   r29.0<2;1,0>:uw   0x10:uw             
        shl (16|M0)              r28.0<1>:ud   r24.0<2;1,0>:uw   0x10:uw              {$5.dst}
        shl (16|M16)             r29.0<1>:ud   r25.0<2;1,0>:uw   0x10:uw             
        shl (16|M0)              r24.0<1>:ud   r20.0<2;1,0>:uw   0x10:uw              {$8.dst}
        shl (16|M16)             r25.0<1>:ud   r21.0<2;1,0>:uw   0x10:uw             
        shl (16|M0)              r20.0<1>:ud   r12.0<2;1,0>:uw   0x10:uw              {$3.dst}
        shl (16|M16)             r21.0<1>:ud   r13.0<2;1,0>:uw   0x10:uw             
L4512:
(W)     add (1|M0)               r4.0<1>:q     r1.6<0;1,0>:q     r2.5<0;1,0>:q    {Compacted}
(W)     send.ugm (1|M0)          r88      r4  null:0  0x0            0x02109580           {I@1,$12} // wr:1+0, rd:1; load.ugm.d32x2t.a64
(W)     cmp (32|M0)   (lt)f0.0   null<1>:d     r88.0<0;1,0>:d    r88.1<0;1,0>:d   {$12.dst}
(W&~f0.0) jmpi                               L13248                                
L4568:
(W)     mov (1|M0)               r1.8<1>:ud    a0.2<0;1,0>:ud                  
(W)     shr (1|M0)               a0.2<1>:ud    r1.9<0;1,0>:ud    0x4:ud             
(W)     mov (1|M0)               r1.10<1>:f    r3.2<0;1,0>:d                   
        cmp (32|M0)   (eq)f3.0   null<2>:w     r52.0<1;1,0>:w    0:w              
(W)     send.ugm (1|M0)          r4       r54  null:0  0x80200000:a0.2        0x4210D500           {$13} // wr:1+0, rd:1; load.ugm.d32x16t.a32.ss[a0.2][A-0xFFC0]
(W)     math.sqt (1|M0)          r54.8<1>:f    r1.10<0;1,0>:f                   {@1,$13.src}
(W)     mov (1|M0)               r1.5<1>:q     r3.0<0;1,0>:d                    {M@1}
(W)     mov (1|M0)               r54.7<1>:ud   0x3F317200:ud                             
(W)     mov (1|M0)               r54.6<1>:ud   0x35BFBE8E:ud                             
(W)     mov (1|M0)               r54.5<1>:f    -0.5:f                              
(W)     mov (1|M0)               r54.4<1>:f    0x3EAAAA83:f                              
(W)     mov (1|M0)               r54.3<1>:f    0xBE7FFF78:f                              
(W)     mov (1|M0)               r3.15<1>:f    0x3E4CE814:f                              
(W)     mov (1|M0)               r3.14<1>:f    0xBE2ACEE6:f                              
(W)     mov (1|M0)               r3.13<1>:f    1.400587e-01:f                              
(W)     mov (1|M0)               r3.12<1>:f    0xBDF9889E:f                              
(W)     mov (1|M0)               r54.1<1>:f    0xBE0402C8:f                              
(W)     mov (1|M0)               r3.5<1>:f     0x3E0F335D:f                              
(W)     mov (1|M0)               r2.7<1>:f     1.0:f                              
(W)     mov (1|M0)               r1.12<1>:d    r88.0<0;1,0>:d                  
(W)     mov (2|M0)               r2.30<1>:w    r4.0<1;1,0>:w                    {$13.dst}
(W)     send.ugm (1|M0)          r4       r54  null:0  0x80000000:a0.2        0x4210D500           {A@1,$14} // wr:1+0, rd:1; load.ugm.d32x16t.a32.ss[a0.2][A-0x10000]
(W)     mov (1|M0)               a0.2<1>:ud    r1.8<0;1,0>:ud                   {$14.src}
(W)     mov (2|M0)               r2.20<1>:w    r4.0<1;1,0>:w                    {$14.dst}
L4952:
(W)     mul (1|M0)               acc0.0<1>:d   r1.12<0;1,0>:d    r3.6<0;1,0>:uw  
(W)     macl (1|M0)              r4.0<1>:d     r1.12<0;1,0>:d    r3.3<0;1,0>:d    {Compacted,$15.src}
(W)     add (1|M0)               r1.14<1>:d    r4.0<0;1,0>:d     r3.7<0;1,0>:d    {I@1}
(W)     shl (1|M0)               r6.1<1>:q     r1.14<0;1,0>:d    1:w               {I@1}
(W)     add (1|M0)               r6.0<1>:q     r6.1<0;1,0>:q     r2.0<0;1,0>:q    {Compacted,I@1}
(W)     add (1|M0)               r4.0<1>:q     r6.1<0;1,0>:q     r2.1<0;1,0>:q    {Compacted}
(W)     send.ugm (1|M0)          r5       r6  null:0  0x0            0x04100B80           {I@2,$0} // wr:2+0, rd:1; load.ugm.d16u32.a64
(W)     send.ugm (1|M0)          r4       r4  null:0  0x0            0x04100B80           {I@1,$0} // wr:2+0, rd:1; load.ugm.d16u32.a64
(W)     shl (1|M0)               r1.13<1>:ud   r5.0<0;1,0>:uw    0x10:uw             
(W)     shl (1|M0)               r1.15<1>:ud   r4.0<0;1,0>:uw    0x10:uw              {$0.dst}
(W)     mul (1|M0)               r54.2<1>:f    r1.13<0;1,0>:f    -1.442695e+00:f               {I@2}
(W)     add (1|M0)               r1.15<1>:f    r1.15<0;1,0>:f    r2.10<0;1,0>:f   {I@1}
(W)     rndz (1|M0)              r4.0<1>:f     r54.2<0;1,0>:f                   {Compacted,F@2}
(W)     mov (1|M0)               r1.30<1>:bf   r1.15<0;1,0>:f                   {F@2}
(W)     mad (1|M0)               r4.2<1>:f     -r1.13<0;0>:f     r3.10<0;0>:f      r4.0<0>:f        {F@2}
(W)     math.exp (1|M0)          r6.0<1>:f     r4.0<0;1,0>:f                   
(W)     mad (1|M0)               r4.2<1>:f     r4.2<0;0>:f       r3.9<0;0>:f       r4.0<0>:f        {F@1}
(W)     cmp (1|M0)    (gt)f2.0   null<1>:f     r1.13<0;1,0>:f    105.0:f              
(W)     mul (1|M0)               r4.1<1>:f     r4.2<0;1,0>:f     1.442695e+00:f               {F@2}
(W)     shl (1|M0)               r1.15<1>:ud   r1.30<0;1,0>:uw   0x10:uw             
(W)     math.exp (1|M0)          r6.1<1>:f     r4.1<0;1,0>:f                    {F@1}
(W)     mad (1|M0)               r54.2<1>:f    r2.7<0;0>:f       r6.0<0;0>:f       r6.1<0>:f        {M@1}
(W)     cmp (1|M0)    (lt)f1.0   null<1>:f     r1.13<0;1,0>:f    -105.0:f              
(W)     math.inv (1|M0)          r54.2<1>:f    r54.2<0;1,0>:f                   {F@2}
(W)     cmp (32|M0)   (lt)f0.0   null<1>:f     r1.15<0;1,0>:f    20.0:f               {I@1}
(W&~f2.0) sel (1|M0)             r54.2<1>:f    r54.2<0;1,0>:f    1.0:f               {M@1}
(W&~f1.0) sel (1|M0)             r2.5<1>:f     r54.2<0;1,0>:f    0.0:f               {F@1}
(W&~f0.0) jmpi                               L6032                                
L5368:
(W)     mul (1|M0)               r54.2<1>:f    r1.15<0;1,0>:f    1.442695e+00:f              
(W)     cmp (1|M0)    (lt)f0.0   null<1>:f     r1.15<0;1,0>:f    -105.0:f               {I@1}
(W)     rndz (1|M0)              r4.0<1>:f     r54.2<0;1,0>:f                   {Compacted,F@2}
(W)     cmp (1|M0)    (gt)f2.0   null<1>:f     r1.15<0;1,0>:f    105.0:f              
(W)     mad (1|M0)               r4.2<1>:f     r1.15<0;0>:f      r3.10<0;0>:f      r4.0<0>:f        {F@2}
(W)     math.exp (1|M0)          r6.0<1>:f     r4.0<0;1,0>:f                   
(W)     mad (1|M0)               r4.2<1>:f     r4.2<0;0>:f       r3.9<0;0>:f       r4.0<0>:f        {F@1}
(W)     mul (1|M0)               r4.1<1>:f     r4.2<0;1,0>:f     1.442695e+00:f               {F@1}
(W)     math.exp (1|M0)          r6.1<1>:f     r4.1<0;1,0>:f                    {F@1}
(W)     mad (1|M0)               r54.2<1>:f    r2.7<0;0>:f       r6.0<0;0>:f       r6.1<0>:f        {M@1}
(W&~f0.0) sel (1|M0)             r54.2<1>:f    r54.2<0;1,0>:f    1.0:f               {F@1}
(W&~f2.0) sel (1|M0)             r54.2<1>:f    r54.2<0;1,0>:f    inf:f               {F@1}
(W)     cmp (32|M0)   (gt)f0.0   null<1>:f     r54.2<0;1,0>:f    0.0:f               {F@1}
(W)     and (1|M0)               r54.9<1>:d    r54.2<0;1,0>:d    2147483647:d              
(W&f0.0) cmp (32|M0)  (lt)f0.0   null<1>:f     r54.9<0;1,0>:f    inf:f               {I@1}
(W&f0.0) jmpi                                L5648                                
L5616:
(W)     math.log (1|M0)          r1.15<1>:f    r54.2<0;1,0>:f                  
(W)     jmpi                                 L6032                                
L5648:
(W)     cmp (32|M0)   (lt)f1.0   null<1>:f     r54.2<0;1,0>:f    0x800000:f              
(W)     mul (1|M0)               r54.10<1>:f   r54.2<0;1,0>:f    8.388608e+06:f              
(W)     mov (1|M0)               r54.9<1>:ud   0xC1B80000:ud                              {F@3}
(W&f1.0) sel (1|M0)              r54.2<1>:f    r54.10<0;1,0>:f   r54.2<0;1,0>:f   {A@1}
(W&f1.0) sel (1|M0)              r54.11<1>:f   r54.9<0;1,0>:f    0.0:f               {I@1}
(W)     add (1|M0)               r54.2<1>:d    r54.2<0;1,0>:d    -1059760811:d               {F@2}
(W)     and (1|M0)               r54.9<1>:d    r54.2<0;1,0>:d    8388607:d               {A@1}
(W)     asr (1|M0)               r54.2<1>:d    r54.2<0;1,0>:d    23:w              
(W)     add (1|M0)               r54.9<1>:d    r54.9<0;1,0>:d    1059760811:d               {I@2}
(W)     mov (1|M0)               r54.2<1>:f    r54.2<0;1,0>:d                   {I@2}
(W)     add (1|M0)               r1.13<1>:f    r54.9<0;1,0>:f    -1.0:f               {I@1}
(W)     add (1|M0)               r54.2<1>:f    r54.11<0;1,0>:f   r54.2<0;1,0>:f   {F@2}
(W)     mad (1|M0)               r54.9<1>:f    r3.5<0;0>:f       r54.1<0;0>:f      r1.13<0>:f       {F@2}
(W)     mad (1|M0)               r54.9<1>:f    r3.12<0;0>:f      r1.13<0;0>:f      r54.9<0>:f       {F@1}
(W)     mad (1|M0)               r54.9<1>:f    r3.13<0;0>:f      r1.13<0;0>:f      r54.9<0>:f       {F@1}
(W)     mad (1|M0)               r54.9<1>:f    r3.14<0;0>:f      r1.13<0;0>:f      r54.9<0>:f       {F@1}
(W)     mad (1|M0)               r54.9<1>:f    r3.15<0;0>:f      r1.13<0;0>:f      r54.9<0>:f       {F@1}
(W)     mad (1|M0)               r54.9<1>:f    r54.3<0;0>:f      r1.13<0;0>:f      r54.9<0>:f       {F@1}
(W)     mad (1|M0)               r54.9<1>:f    r54.4<0;0>:f      r1.13<0;0>:f      r54.9<0>:f       {F@1}
(W)     mad (1|M0)               r54.9<1>:f    r54.5<0;0>:f      r1.13<0;0>:f      r54.9<0>:f       {F@1}
(W)     mul (1|M0)               r54.9<1>:f    r1.13<0;1,0>:f    r54.9<0;1,0>:f   {F@1}
(W)     mad (1|M0)               r54.9<1>:f    r1.13<0;0>:f      r1.13<0;0>:f      r54.9<0>:f       {F@1}
(W)     mad (1|M0)               r54.9<1>:f    r54.9<0;0>:f      r54.6<0;0>:f      r54.2<0>:f       {F@1}
(W)     mad (1|M0)               r1.15<1>:f    r54.9<0;0>:f      r54.7<0;0>:f      r54.2<0>:f       {F@1}
L6032:
(W)     mul (1|M0)               r12.1<1>:f    r1.15<0;1,0>:f    -r2.15<0;1,0>:f  {F@1}
(W)     cmp (32|M0)   (eq)f1.0   null<1>:d     r3.8<0;1,0>:d     0:w              
(W)     mul (1|M0)               r1.13<1>:f    r12.1<0;1,0>:f    1.442695e+00:f               {F@1}
(W)     mul (1|M0)               acc0.0<1>:d   r1.12<0;1,0>:d    r3.2<0;1,0>:uw  
(W)     rndz (1|M0)              r1.13<1>:f    r1.13<0;1,0>:f                   {F@1}
(W)     macl (1|M0)              r5.0<1>:d     r1.12<0;1,0>:d    r3.1<0;1,0>:d    {Compacted}
(W)     mad (1|M0)               r12.0<1>:f    r12.1<0;0>:f      r3.10<0;0>:f      r1.13<0>:f       {F@1}
(W)     mad (1|M0)               r1.15<1>:f    r12.0<0;0>:f      r3.9<0;0>:f       r1.13<0>:f       {F@1}
(W)     mul (1|M0)               r1.15<1>:f    r1.15<0;1,0>:f    1.442695e+00:f               {F@1}
(W&~f1.0) jmpi                               L6216                                
L6184:
(W)     mov (1|M0)               r12.6<1>:d    -1:w                              
(W)     jmpi                                 L6656                                
L6216:
(W)     asr (1|M0)               r54.9<1>:d    r3.8<0;1,0>:d     31:w              
(W)     add (1|M0)               r12.0<1>:d    r54.9<0;1,0>:d    r3.8<0;1,0>:d    {Compacted,I@1}
(W)     xor (1|M0)               r54.2<1>:d    r12.0<0;1,0>:d    r54.9<0;1,0>:d   {Compacted,I@1}
(W)     xor (1|M0)               cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x30:uw              {A@1}
(W)     mov (1|M0)               r12.2<1>:f    r54.2<0;1,0>:ud                  {A@1}
(W)     mov (1|M0)               r12.4<1>:f    0xB4C00000:f                               {Compacted}
(W)     math.inv (1|M0)          r12.3<1>:f    r12.2<0;1,0>:f                   {F@2}
(W)     mov (1|M0)               r12.0<1>:ud   r12.2<0;1,0>:f                  
(W)     mad (1|M0)               r12.5<1>:f    r12.3<0;0>:f      r12.4<0;0>:f      r12.3<0>:f       {A@1}
(W)     add (1|M0)               r88.4<1>:d    r54.2<0;1,0>:d    -r12.0<0;1,0>:d  {I@1}
(W)     mov (1|M0)               r12.0<1>:f    r3.7<0;1,0>:ud                   {I@1}
(W)     mov (1|M0)               r54.12<1>:f   r88.4<0;1,0>:ud                 
(W)     mul (1|M0)               r12.4<1>:f    r12.0<0;1,0>:f    r12.5<0;1,0>:f   {Compacted,F@2}
(W)     mov (1|M0)               r12.3<1>:ud   r12.0<0;1,0>:f                  
(W)     mov (1|M0)               r12.4<1>:ud   r12.4<0;1,0>:f                   {F@1}
(W)     add (1|M0)               r88.5<1>:d    r3.7<0;1,0>:d     -r12.3<0;1,0>:d  {I@2}
(W)     mov (1|M0)               r12.3<1>:f    r12.4<0;1,0>:ud                  {I@1}
(W)     mov (1|M0)               r54.13<1>:f   r88.5<0;1,0>:ud                 
(W)     mad (1|M0)               r12.2<1>:f    r12.0<0;0>:f      r12.3<0;0>:f      -r12.2<0>:f      {F@2}
(W)     mad (1|M0)               r12.0<1>:f    r54.13<0;0>:f     r12.3<0;0>:f      -r54.12<0>:f     {F@2}
(W)     add (1|M0)               r12.0<1>:f    r12.2<0;1,0>:f    r12.0<0;1,0>:f   {Compacted,F@1}
(W)     mul (1|M0)               r12.0<1>:f    r12.5<0;1,0>:f    r12.0<0;1,0>:f   {F@1}
(W)     xor (1|M0)               cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x30:uw              {A@1}
(W)     mov (1|M0)               r12.0<1>:ud   r12.0<0;1,0>:f                   {A@1}
(W)     add (1|M0)               r12.0<1>:d    r12.0<0;1,0>:d    r12.4<0;1,0>:d   {Compacted,I@1}
(W)     mul (1|M0)               acc0.0<1>:d   r12.0<0;1,0>:d    r54.4<0;1,0>:uw  {I@1}
(W)     macl (1|M0)              r4.0<1>:d     r12.0<0;1,0>:d    r54.2<0;1,0>:d   {Compacted}
(W)     add (1|M0)               r12.2<1>:d    r3.7<0;1,0>:d     -r4.0<0;1,0>:d   {I@1}
(W)     cmp (1|M0)    (ge)f0.0   r12.2<1>:ud   r12.2<0;1,0>:ud   r54.2<0;1,0>:ud  {I@1}
(W)     add3 (1|M0)              r12.0<1>:d    r12.0<0;0>:d      r54.9<0;0>:d      -r12.2<0>:d      {I@1}
(W)     xor (1|M0)               r12.6<1>:d    r12.0<0;1,0>:d    r54.9<0;1,0>:d   {I@1}
L6656:
(W)     add (1|M0)               r12.0<1>:d    r5.0<0;1,0>:d     r12.6<0;1,0>:d   {Compacted,I@1}
        or (32|M0)               r38.0<1>:d    r16.0<1;1,0>:d    1:w               {Compacted}
(W)     mul (1|M0)               acc0.0<1>:d   r12.0<0;1,0>:d    r3.4<0;1,0>:uw   {I@2}
(W)     macl (1|M0)              r12.0<1>:d    r12.0<0;1,0>:d    r3.2<0;1,0>:d    {Compacted}
        or (32|M0)               r70.0<1>:d    r16.0<1;1,0>:d    2:w               {Compacted}
        add (32|M0)              r6.0<1>:d     r12.0<0;1,0>:d    r38.0<1;1,0>:d   {Compacted,I@2}
        or (32|M0)               r72.0<1>:d    r16.0<1;1,0>:d    3:w               {Compacted}
        add (32|M0)              r8.0<1>:d     r12.0<0;1,0>:d    r16.0<1;1,0>:d   {Compacted}
        add (32|M0)              r4.0<1>:d     r12.0<0;1,0>:d    r70.0<1;1,0>:d   {Compacted,I@4}
        mov (16|M0)              r14.0<2>:ud   r6.0<1;1,0>:ud                   {Compacted,I@4}
        mov (16|M16)             r36.0<2>:ud   r7.0<1;1,0>:ud                   {Compacted}
        add (32|M0)              r10.0<1>:d    r12.0<0;1,0>:d    r72.0<1;1,0>:d   {Compacted,I@5}
(W)     cmp (1|M0)    (lt)f2.0   null<1>:f     r12.1<0;1,0>:f    -105.0:f              
(W)     cmp (1|M0)    (gt)f1.0   null<1>:f     r12.1<0;1,0>:f    105.0:f              
        mov (16|M0)              r40.0<2>:ud   r8.0<1;1,0>:ud                   {Compacted,I@5}
        mov (16|M16)             r12.0<2>:ud   r9.0<1;1,0>:ud                   {Compacted,F@1}
        shl (16|M0)              r42.0<1>:q    r14.0<2;1,0>:d    1:w               {I@5}
        mov (16|M0)              r6.0<2>:ud    r4.0<1;1,0>:ud                   {Compacted}
        shl (16|M16)             r14.0<1>:q    r36.0<2;1,0>:d    1:w               {I@6}
        shl (16|M0)              r44.0<1>:q    r40.0<2;1,0>:d    1:w               {I@5}
        shl (16|M16)             r78.0<1>:q    r12.0<2;1,0>:d    1:w               {I@5}
        mov (16|M16)             r8.0<2>:ud    r5.0<1;1,0>:ud                   {Compacted}
        shl (16|M0)              r46.0<1>:q    r6.0<2;1,0>:d     1:w               {I@5}
        add (16|M0)              r4.0<1>:q     r42.0<1;1,0>:q    r1.2<0;1,0>:q    {Compacted}
        add (16|M16)             r6.0<1>:q     r14.0<1;1,0>:q    r1.2<0;1,0>:q    {Compacted,I@6}
        mov (16|M0)              r40.0<2>:ud   r10.0<1;1,0>:ud                  {Compacted}
        add (16|M16)             r12.0<1>:q    r78.0<1;1,0>:q    r1.2<0;1,0>:q    {Compacted,I@6}
        shl (16|M16)             r76.0<1>:q    r8.0<2;1,0>:d     1:w               {I@6}
        mov (16|M16)             r8.0<2>:ud    r11.0<1;1,0>:ud                  {Compacted}
        add (16|M0)              r10.0<1>:q    r44.0<1;1,0>:q    r1.2<0;1,0>:q    {Compacted}
        send.ugm (32|M0)         r36      r4  null:0  0x0            0x08200B80           {I@6,$1} // wr:4+0, rd:2; load.ugm.d16u32.a64
        shl (16|M0)              r52.0<1>:q    r40.0<2;1,0>:d    1:w               {I@5}
        send.ugm (32|M0)         r40      r10  null:0  0x0            0x08200B80           {I@1,$2} // wr:4+0, rd:2; load.ugm.d16u32.a64
        shl (16|M16)             r74.0<1>:q    r8.0<2;1,0>:d     1:w              
        add (16|M0)              r6.0<1>:q     r46.0<1;1,0>:q    r1.2<0;1,0>:q    {Compacted,$1.src}
        add (16|M16)             r8.0<1>:q     r76.0<1;1,0>:q    r1.2<0;1,0>:q    {Compacted}
        add (16|M0)              r4.0<1>:q     r42.0<1;1,0>:q    r1.1<0;1,0>:q    {Compacted}
        send.ugm (32|M0)         r42      r6  null:0  0x0            0x08200B80           {I@1,$3} // wr:4+0, rd:2; load.ugm.d16u32.a64
        add (16|M0)              r10.0<1>:q    r52.0<1;1,0>:q    r1.2<0;1,0>:q    {Compacted,$2.src}
        add (16|M16)             r12.0<1>:q    r74.0<1;1,0>:q    r1.2<0;1,0>:q    {Compacted}
        add (16|M16)             r6.0<1>:q     r14.0<1;1,0>:q    r1.1<0;1,0>:q    {Compacted,$3.src}
        add (16|M0)              r8.0<1>:q     r44.0<1;1,0>:q    r1.1<0;1,0>:q    {Compacted}
        send.ugm (32|M0)         r44      r10  null:0  0x0            0x08200B80           {I@1,$4} // wr:4+0, rd:2; load.ugm.d16u32.a64
        add (16|M16)             r10.0<1>:q    r78.0<1;1,0>:q    r1.1<0;1,0>:q    {Compacted,$4.src}
        add (16|M0)              r12.0<1>:q    r46.0<1;1,0>:q    r1.1<0;1,0>:q    {Compacted}
        send.ugm (32|M0)         r46      r4  null:0  0x0            0x08200B80           {I@1,$5} // wr:4+0, rd:2; load.ugm.d16u32.a64
        add (16|M0)              r4.0<1>:q     r52.0<1;1,0>:q    r1.1<0;1,0>:q    {Compacted,$5.src}
        send.ugm (32|M0)         r52      r8  null:0  0x0            0x08200B80           {I@1,$6} // wr:4+0, rd:2; load.ugm.d16u32.a64
        add (16|M16)             r14.0<1>:q    r76.0<1;1,0>:q    r1.1<0;1,0>:q    {Compacted}
        send.ugm (32|M0)         r8       r12  null:0  0x0            0x08200B80           {I@1,$7} // wr:4+0, rd:2; load.ugm.d16u32.a64
        add (16|M16)             r6.0<1>:q     r74.0<1;1,0>:q    r1.1<0;1,0>:q    {Compacted}
        send.ugm (32|M0)         r10      r4  null:0  0x0            0x08200B80           {I@1,$8} // wr:4+0, rd:2; load.ugm.d16u32.a64
(W)     math.exp (1|M0)          r54.2<1>:f    r1.13<0;1,0>:f                  
        shl (16|M16)             r78.0<1>:ud   r37.0<2;1,0>:uw   0x10:uw              {$1.dst}
(W)     math.exp (1|M0)          r1.13<1>:f    r1.15<0;1,0>:f                  
(W)     mul (1|M0)               r1.13<1>:f    r54.2<0;1,0>:f    r1.13<0;1,0>:f   {M@1}
        mov (16|M0)              r13.0<1>:uw   r36.0<2;1,0>:uw                  {$7.src}
(W&~f2.0) sel (1|M0)             r1.13<1>:f    r1.13<0;1,0>:f    0.0:f               {F@1}
        shl (16|M0)              r4.0<1>:ud    r36.0<2;1,0>:uw   0x10:uw              {$8.src}
(W&~f1.0) sel (1|M0)             r2.4<1>:f     r1.13<0;1,0>:f    inf:f               {F@1}
        mov (16|M16)             r5.0<1>:uw    r37.0<2;1,0>:uw                 
        shl (16|M0)              r12.0<1>:ud   r40.0<2;1,0>:uw   0x10:uw              {$2.dst}
        mul (32|M0)              acc2.0<1>:f   r58.0<1;1,0>:f    r2.4<0;1,0>:f    {Compacted,F@1}
        mov (16|M0)              r37.0<1>:uw   r40.0<2;1,0>:uw                 
        mul (16|M0)              acc0.0<1>:f   r13.0<1;1,0>:bf   r4.0<1;1,0>:f    {I@4}
        mul (16|M16)             acc1.0<1>:f   r5.0<1;1,0>:bf    r78.0<1;1,0>:f   {I@3}
        mov (16|M16)             r79.0<1>:uw   r41.0<2;1,0>:uw                 
        shl (16|M16)             r36.0<1>:ud   r41.0<2;1,0>:uw   0x10:uw             
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     r37.0<1;0>:bf     r12.0<1>:f       {I@3}
        sync.nop                             null                             {Compacted,F@3}
        shl (16|M0)              r4.0<1>:ud    r42.0<2;1,0>:uw   0x10:uw              {$3.dst}
        mov (16|M0)              r78.0<1>:uw   r42.0<2;1,0>:uw                  {F@2}
        mad (16|M16)             acc1.0<1>:f   acc1.0<1;0>:f     r79.0<1;0>:bf     r36.0<1>:f       {I@3}
        shl (16|M16)             r40.0<1>:ud   r43.0<2;1,0>:uw   0x10:uw             
        mov (16|M16)             r42.0<1>:uw   r43.0<2;1,0>:uw                 
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     r78.0<1;0>:bf     r4.0<1>:f        {I@3}
        sync.nop                             null                             {Compacted,F@2}
        shl (16|M16)             r36.0<1>:ud   r45.0<2;1,0>:uw   0x10:uw              {$4.dst}
        mov (16|M16)             r43.0<1>:uw   r45.0<2;1,0>:uw                 
        mad (16|M16)             acc1.0<1>:f   acc1.0<1;0>:f     r42.0<1;0>:bf     r40.0<1>:f       {I@3}
(W)     mov (32|M0)              r48.0<1>:ud   0x0:ud                             
        mov (16|M0)              r80.0<1>:uw   r44.0<2;1,0>:uw                 
        mov (16|M0)              r41.0<1>:uw   r46.0<2;1,0>:uw                  {$5.dst}
        shl (16|M0)              r12.0<1>:ud   r44.0<2;1,0>:uw   0x10:uw             
        shl (16|M0)              r4.0<1>:ud    r46.0<2;1,0>:uw   0x10:uw              {F@2}
        mov (16|M16)             r40.0<1>:uw   r47.0<2;1,0>:uw                  {F@1}
        shl (16|M16)             r44.0<1>:ud   r47.0<2;1,0>:uw   0x10:uw             
        mad (16|M16)             r49.0<1>:f    acc1.0<1;0>:f     r43.0<1;0>:bf     r36.0<1>:f       {I@7}
        mov (16|M0)              r46.0<1>:uw   r52.0<2;1,0>:uw                  {$6.dst}
        shl (16|M0)              r36.0<1>:ud   r52.0<2;1,0>:uw   0x10:uw              {F@1}
        mad (16|M0)              r48.0<1>:f    acc0.0<1;0>:f     r80.0<1;0>:bf     r12.0<1>:f       {I@6}
        mul (16|M0)              acc0.0<1>:f   r41.0<1;1,0>:bf   r4.0<1;1,0>:f    {I@5}
        mul (16|M16)             acc1.0<1>:f   r40.0<1;1,0>:bf   r44.0<1;1,0>:f   {I@3}
        shl (16|M16)             r84.0<1>:ud   r53.0<2;1,0>:uw   0x10:uw             
        mov (16|M16)             r45.0<1>:uw   r53.0<2;1,0>:uw                 
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     r46.0<1;0>:bf     r36.0<1>:f       {I@3}
(W)     add (16|M0)              r48.0<1>:f    r48.0<1;1,0>:f    r49.0<1;1,0>:f   {Compacted,F@4}
        shl (16|M0)              r12.0<1>:ud   r8.0<2;1,0>:uw    0x10:uw              {$7.dst}
        mov (16|M0)              r44.0<1>:uw   r8.0<2;1,0>:uw                   {F@3}
        mad (16|M16)             acc1.0<1>:f   acc1.0<1;0>:f     r45.0<1;0>:bf     r84.0<1>:f       {I@3}
        mov (16|M16)             r47.0<1>:uw   r9.0<2;1,0>:uw                  
        shl (16|M16)             r52.0<1>:ud   r9.0<2;1,0>:uw    0x10:uw             
(W)     mov (8|M0)               r49.0<1>:ud   r48.8<1;1,0>:ud                  {Compacted,F@2}
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     r44.0<1;0>:bf     r12.0<1>:f       {I@4}
        shl (16|M0)              r4.0<1>:ud    r10.0<2;1,0>:uw   0x10:uw              {$8.dst}
        shl (16|M16)             r8.0<1>:ud    r11.0<2;1,0>:uw   0x10:uw             
(W)     add (8|M0)               r9.0<1>:f     r48.0<1;1,0>:f    r49.0<1;1,0>:f   {Compacted,I@3}
        mad (16|M16)             acc1.0<1>:f   acc1.0<1;0>:f     r47.0<1;0>:bf     r52.0<1>:f      
        mov (16|M0)              r49.0<1>:uw   r10.0<2;1,0>:uw                  {F@2}
        mov (16|M16)             r48.0<1>:uw   r11.0<2;1,0>:uw                 
(W)     mov (32|M0)              r50.0<1>:ud   0x0:ud                             
(W)     mov (4|M0)               r10.0<1>:ud   r9.4<1;1,0>:ud                   {Compacted}
        mad (16|M0)              r50.0<1>:f    acc0.0<1;0>:f     r49.0<1;0>:bf     r4.0<1>:f        {I@2}
        mad (16|M16)             r51.0<1>:f    acc1.0<1;0>:f     r48.0<1;0>:bf     r8.0<1>:f       
(W)     add (4|M0)               r8.0<1>:f     r9.0<1;1,0>:f     r10.0<1;1,0>:f   {Compacted,I@1}
(W)     add (16|M0)              r4.0<1>:f     r50.0<1;1,0>:f    r51.0<1;1,0>:f   {Compacted,F@2}
(W)     add (1|M0)               r5.8<1>:f     r8.0<0;1,0>:f     r8.2<0;1,0>:f    {F@2}
(W)     add (1|M0)               r5.9<1>:f     r8.1<0;1,0>:f     r8.3<0;1,0>:f   
(W)     mov (8|M0)               r8.0<1>:ud    r4.8<1;1,0>:ud                   {Compacted,F@1}
(W)     add (1|M0)               r40.9<1>:f    r5.8<0;1,0>:f     r5.9<0;1,0>:f   
(W)     add (8|M0)               r4.0<1>:f     r4.0<1;1,0>:f     r8.0<1;1,0>:f    {Compacted,I@1}
(W)     add (1|M0)               r40.13<1>:f   r40.9<0;1,0>:f    1e-06:f               {F@2}
(W)     mov (4|M0)               r8.0<1>:ud    r4.4<1;1,0>:ud                   {Compacted,F@2}
(W)     math.rsqt (1|M0)         r1.13<1>:f    r40.13<0;1,0>:f                  {F@1}
(W)     add (4|M0)               r10.0<1>:f    r4.0<1;1,0>:f     r8.0<1;1,0>:f    {Compacted,I@1}
        mul (32|M0)              r6.0<1>:f     r66.0<1;1,0>:f    r2.4<0;1,0>:f    {Compacted}
(W)     add (1|M0)               r10.4<1>:f    r10.0<0;1,0>:f    r10.2<0;1,0>:f   {Compacted,F@2}
(W)     add (1|M0)               r10.5<1>:f    r10.1<0;1,0>:f    r10.3<0;1,0>:f  
        mul (16|M0)              r12.0<1>:f    r13.0<1;1,0>:bf   r1.13<0;1,0>:f   {M@1}
(W)     add (1|M0)               r40.8<1>:f    r10.4<0;1,0>:f    r10.5<0;1,0>:f   {F@2}
        mul (16|M16)             r13.0<1>:f    r5.0<1;1,0>:bf    r1.13<0;1,0>:f  
(W)     add (1|M0)               r40.12<1>:f   r40.8<0;1,0>:f    1e-06:f               {F@2}
        mul (32|M0)              acc0.0<1>:f   r6.0<1;1,0>:f     r12.0<1;1,0>:f   {Compacted,F@2}
        mul (32|M0)              r74.0<1>:f    r68.0<1;1,0>:f    r2.4<0;1,0>:f    {Compacted}
        mul (16|M0)              r36.0<1>:f    r37.0<1;1,0>:bf   r1.13<0;1,0>:f  
        mul (16|M16)             r9.0<1>:f     r42.0<1;1,0>:bf   r1.13<0;1,0>:f  
        mul (16|M0)              r8.0<1>:f     r78.0<1;1,0>:bf   r1.13<0;1,0>:f  
        mul (16|M0)              r4.0<1>:f     r80.0<1;1,0>:bf   r1.13<0;1,0>:f  
        mul (16|M16)             r5.0<1>:f     r43.0<1;1,0>:bf   r1.13<0;1,0>:f  
        mul (32|M0)              acc2.0<1>:f   acc2.0<1;1,0>:f   r12.0<1;1,0>:f   {Compacted}
        mul (16|M16)             r37.0<1>:f    r79.0<1;1,0>:bf   r1.13<0;1,0>:f  
(W)     math.sqt (1|M0)          r1.13<1>:f    r40.12<0;1,0>:f                  {F@1}
        mul (32|M0)              r14.0<1>:f    r60.0<1;1,0>:f    r2.4<0;1,0>:f    {Compacted}
        mul (32|M0)              r76.0<1>:f    r64.0<1;1,0>:f    r2.4<0;1,0>:f    {Compacted}
        mad (32|M0)              acc0.0<1>:f   acc0.0<1;0>:f     r74.0<1;0>:f      r36.0<1>:f       {Compacted}
(W)     mul (1|M0)               r1.13<1>:f    r54.8<0;1,0>:f    r1.13<0;1,0>:f   {M@1}
        mad (32|M0)              r52.0<1>:f    acc2.0<1;0>:f     r14.0<1;0>:f      r36.0<1>:f       {Compacted,F@4}
        mad (32|M0)              r80.0<1>:f    acc0.0<1;0>:f     r76.0<1;0>:f      r8.0<1>:f        {Compacted,F@4}
(W)     mul (1|M0)               acc0.0<1>:d   r1.14<0;1,0>:d    r3.8<0;1,0>:uw   {F@1}
        mul (32|M0)              acc2.0<1>:f   r30.0<1;1,0>:f    r2.4<0;1,0>:f    {Compacted}
(W)     math.inv (1|M0)          r1.13<1>:f    r1.13<0;1,0>:f                  
(W)     macl (1|M0)              r42.0<1>:d    r1.14<0;1,0>:d    r3.4<0;1,0>:d    {Compacted}
        mul (32|M0)              acc0.0<1>:f   r22.0<1;1,0>:f    r2.4<0;1,0>:f    {Compacted,I@1}
        mul (32|M0)              acc2.0<1>:f   acc2.0<1;1,0>:f   r12.0<1;1,0>:f   {Compacted}
        mul (16|M0)              r14.0<1>:f    r41.0<1;1,0>:bf   r1.13<0;1,0>:f   {M@1}
        mul (16|M16)             r15.0<1>:f    r40.0<1;1,0>:bf   r1.13<0;1,0>:f  
        mul (16|M0)              r10.0<1>:f    r44.0<1;1,0>:bf   r1.13<0;1,0>:f  
        mul (16|M16)             r11.0<1>:f    r47.0<1;1,0>:bf   r1.13<0;1,0>:f  
        mul (16|M0)              r6.0<1>:f     r49.0<1;1,0>:bf   r1.13<0;1,0>:f  
        mul (16|M16)             r7.0<1>:f     r48.0<1;1,0>:bf   r1.13<0;1,0>:f  
        mul (16|M16)             r41.0<1>:f    r45.0<1;1,0>:bf   r1.13<0;1,0>:f  
        mul (16|M0)              r40.0<1>:f    r46.0<1;1,0>:bf   r1.13<0;1,0>:f  
        mul (32|M0)              r44.0<1>:f    r32.0<1;1,0>:f    r2.4<0;1,0>:f    {Compacted}
(W)     add (1|M0)               r1.13<1>:d    r42.0<0;1,0>:d    r3.6<0;1,0>:d    {F@2}
        mul (32|M0)              acc0.0<1>:f   acc0.0<1;1,0>:f   r12.0<1;1,0>:f   {Compacted}
        mul (32|M0)              r42.0<1>:f    r24.0<1;1,0>:f    r2.4<0;1,0>:f    {Compacted,I@1}
(W)     shl (1|M0)               r1.7<1>:q     r1.13<0;1,0>:d    1:w              
        mad (32|M0)              acc2.0<1>:f   acc2.0<1;0>:f     r44.0<1;0>:f      r36.0<1>:f       {Compacted,F@3}
(W)     mov (2|M0)               r88.16<1>:w   0x40:uv                             
(W)     add (1|M0)               r44.0<1>:q    r1.7<0;1,0>:q     r1.3<0;1,0>:q    {Compacted,A@1}
        mad (32|M0)              acc0.0<1>:f   acc0.0<1;0>:f     r42.0<1;0>:f      r36.0<1>:f       {Compacted}
(W)     add (1|M0)               r42.0<1>:uq   r44.0<0;1,0>:uq   r88.16<0;1,0>:w  {A@1}
(W)     add (1|M0)               r42.1<1>:uq   r44.0<0;1,0>:uq   r88.17<0;1,0>:w 
        mul (32|M0)              r48.0<1>:f    r56.0<1;1,0>:f    r2.4<0;1,0>:f    {Compacted}
(W)     send.ugm (2|M0)          r84      r42  null:0  0x0            0x04100580           {I@1,$9} // wr:2+0, rd:1; load.ugm.d32.a64
        mul (32|M0)              r46.0<1>:f    r28.0<1;1,0>:f    r2.4<0;1,0>:f    {Compacted}
        mul (32|M0)              r82.0<1>:f    r62.0<1;1,0>:f    r2.4<0;1,0>:f    {Compacted}
        mul (32|M0)              r50.0<1>:f    r20.0<1;1,0>:f    r2.4<0;1,0>:f    {Compacted}
        mul (32|M0)              r78.0<1>:f    r34.0<1;1,0>:f    r2.4<0;1,0>:f    {Compacted}
        mad (32|M0)              r52.0<1>:f    r52.0<1;0>:f      r48.0<1;0>:f      r8.0<1>:f        {Compacted,F@5}
        mad (32|M0)              acc2.0<1>:f   acc2.0<1;0>:f     r46.0<1;0>:f      r8.0<1>:f        {Compacted,F@5}
        mul (32|M0)              r74.0<1>:f    r18.0<1;1,0>:f    r2.4<0;1,0>:f    {Compacted}
        mul (32|M0)              r76.0<1>:f    r26.0<1;1,0>:f    r2.4<0;1,0>:f    {Compacted}
(W)     mov (32|M0)              r48.0<1>:ud   0x0:ud                              {F@4}
        mad (32|M0)              acc0.0<1>:f   acc0.0<1;0>:f     r50.0<1;0>:f      r8.0<1>:f        {Compacted}
(W)     mov (32|M0)              r46.0<1>:ud   0x0:ud                              {F@4}
        mad (32|M0)              r48.0<1>:f    r80.0<1;0>:f      r82.0<1;0>:f      r4.0<1>:f        {Compacted,I@2}
(W)     mov (32|M0)              r44.0<1>:ud   0x0:ud                             
(W)     mov (32|M0)              r42.0<1>:ud   0x0:ud                              {$9.src}
        mad (32|M0)              r46.0<1>:f    r52.0<1;0>:f      r78.0<1;0>:f      r4.0<1>:f        {Compacted,I@3}
        mad (32|M0)              r44.0<1>:f    acc2.0<1;0>:f     r76.0<1;0>:f      r4.0<1>:f        {Compacted,A@2}
        mad (32|M0)              r42.0<1>:f    acc0.0<1;0>:f     r74.0<1;0>:f      r4.0<1>:f        {Compacted,I@1}
(W)     add (16|M0)              r53.0<1>:f    r48.0<1;1,0>:f    r49.0<1;1,0>:f   {Compacted,F@4}
(W)     add (16|M0)              r51.0<1>:f    r46.0<1;1,0>:f    r47.0<1;1,0>:f   {Compacted,F@4}
(W)     add (16|M0)              r52.0<1>:f    r44.0<1;1,0>:f    r45.0<1;1,0>:f   {Compacted,F@4}
(W)     add (16|M0)              r50.0<1>:f    r42.0<1;1,0>:f    r43.0<1;1,0>:f   {Compacted,F@4}
(W)     mov (8|M0)               r42.0<1>:ud   r53.8<1;1,0>:ud                  {Compacted,F@1}
(W)     mov (8|M0)               r43.0<1>:ud   r51.8<1;1,0>:ud                  {Compacted}
(W)     mov (8|M0)               r75.0<1>:ud   r52.8<1;1,0>:ud                  {Compacted}
(W)     mov (8|M0)               r76.0<1>:ud   r50.8<1;1,0>:ud                  {Compacted}
(W)     add (8|M0)               r74.0<1>:f    r53.0<1;1,0>:f    r42.0<1;1,0>:f   {Compacted,I@4}
(W)     add (8|M0)               r53.0<1>:f    r51.0<1;1,0>:f    r43.0<1;1,0>:f   {Compacted,I@3}
(W)     add (8|M0)               r50.0<1>:f    r50.0<1;1,0>:f    r76.0<1;1,0>:f   {Compacted,I@1}
(W)     add (8|M0)               r51.0<1>:f    r52.0<1;1,0>:f    r75.0<1;1,0>:f   {Compacted}
(W)     mov (4|M0)               r52.0<1>:ud   r74.4<1;1,0>:ud                  {Compacted,F@1}
(W)     mov (4|M0)               r75.0<1>:ud   r53.4<1;1,0>:ud                  {Compacted}
(W)     mov (4|M0)               r77.0<1>:ud   r50.4<1;1,0>:ud                  {Compacted}
(W)     mov (4|M0)               r76.0<1>:ud   r51.4<1;1,0>:ud                  {Compacted}
(W)     add (4|M0)               r74.0<1>:f    r74.0<1;1,0>:f    r52.0<1;1,0>:f   {Compacted,I@4}
(W)     add (4|M0)               r52.0<1>:f    r53.0<1;1,0>:f    r75.0<1;1,0>:f   {Compacted,I@3}
(W)     add (4|M0)               r50.0<1>:f    r50.0<1;1,0>:f    r77.0<1;1,0>:f   {Compacted,I@2}
(W)     add (4|M0)               r51.0<1>:f    r51.0<1;1,0>:f    r76.0<1;1,0>:f   {Compacted,I@1}
(W)     add (1|M0)               r50.10<1>:f   r74.0<0;1,0>:f    r74.2<0;1,0>:f   {F@4}
(W)     add (1|M0)               r50.11<1>:f   r74.1<0;1,0>:f    r74.3<0;1,0>:f  
(W)     add (1|M0)               r50.8<1>:f    r52.0<0;1,0>:f    r52.2<0;1,0>:f   {F@5}
(W)     add (1|M0)               r50.9<1>:f    r52.1<0;1,0>:f    r52.3<0;1,0>:f  
(W)     shl (1|M0)               r54.2<1>:ud   r84.0<0;1,0>:uw   0x10:uw              {$9.dst}
(W)     add (1|M0)               r50.4<1>:f    r50.0<0;1,0>:f    r50.2<0;1,0>:f   {Compacted,F@6}
(W)     add (1|M0)               r50.6<1>:f    r51.0<0;1,0>:f    r51.2<0;1,0>:f   {F@6}
(W)     add (1|M0)               r50.7<1>:f    r51.1<0;1,0>:f    r51.3<0;1,0>:f  
(W)     add (1|M0)               r50.2<1>:f    r50.10<0;1,0>:f   r50.11<0;1,0>:f  {F@6}
(W)     shl (1|M0)               r54.9<1>:ud   r84.1<0;1,0>:uw   0x10:uw             
(W)     add (1|M0)               r50.5<1>:f    r50.1<0;1,0>:f    r50.3<0;1,0>:f  
(W)     add (1|M0)               r50.1<1>:f    r50.8<0;1,0>:f    r50.9<0;1,0>:f   {F@6}
(W)     shl (1|M0)               r54.13<1>:ud  r84.2<0;1,0>:uw   0x10:uw             
(W)     add (1|M0)               r50.0<1>:f    r50.6<0;1,0>:f    r50.7<0;1,0>:f   {Compacted,F@4}
(W)     add (1|M0)               r54.11<1>:f   r54.2<0;1,0>:f    -r50.2<0;1,0>:f  {A@3}
(W)     shl (1|M0)               r54.12<1>:ud  r84.3<0;1,0>:uw   0x10:uw             
(W)     add (1|M0)               r54.14<1>:f   r50.4<0;1,0>:f    r50.5<0;1,0>:f   {F@4}
(W)     add (1|M0)               r54.10<1>:f   r54.9<0;1,0>:f    -r50.1<0;1,0>:f  {A@3}
(W)     mul (1|M0)               r3.11<1>:f    r54.11<0;1,0>:f   r2.5<0;1,0>:f    {F@3}
(W)     add (1|M0)               r54.9<1>:f    r54.13<0;1,0>:f   -r50.0<0;1,0>:f  {I@2}
(W)     add (1|M0)               r54.2<1>:f    r54.12<0;1,0>:f   -r54.14<0;1,0>:f {A@1}
(W)     mul (1|M0)               r3.0<1>:f     r54.10<0;1,0>:f   r2.5<0;1,0>:f    {Compacted,F@4}
        mul (32|M0)              acc0.0<1>:f   r12.0<1;1,0>:f    r3.11<0;1,0>:f   {Compacted,F@4}
(W)     mul (1|M0)               r2.11<1>:f    r54.9<0;1,0>:f    r2.5<0;1,0>:f    {F@4}
        mul (32|M0)              acc2.0<1>:f   r12.0<1;1,0>:f    r3.0<0;1,0>:f    {Compacted,F@3}
(W)     mul (1|M0)               r2.5<1>:f     r54.2<0;1,0>:f    r2.5<0;1,0>:f   
        mul (32|M0)              r50.0<1>:f    r12.0<1;1,0>:f    r2.11<0;1,0>:f   {Compacted,F@3}
        mul (32|M0)              r74.0<1>:f    r36.0<1;1,0>:f    r3.11<0;1,0>:f   {Compacted}
        mad (32|M0)              r66.0<1>:f    acc0.0<1;0>:f     r66.0<1;0>:f      r2.4<0>:f        {Compacted}
        mad (32|M0)              r58.0<1>:f    acc2.0<1;0>:f     r58.0<1;0>:f      r2.4<0>:f        {Compacted}
        mul (32|M0)              r12.0<1>:f    r12.0<1;1,0>:f    r2.5<0;1,0>:f    {Compacted,F@5}
        mul (32|M0)              r52.0<1>:f    r36.0<1;1,0>:f    r3.0<0;1,0>:f    {Compacted}
        mul (32|M0)              acc2.0<1>:f   r36.0<1;1,0>:f    r2.11<0;1,0>:f   {Compacted}
        mad (32|M0)              r30.0<1>:f    r50.0<1;0>:f      r30.0<1;0>:f      r2.4<0>:f        {Compacted,F@7}
        mul (32|M0)              r84.0<1>:f    r66.0<1;1,0>:f    r14.0<1;1,0>:f   {Compacted,F@6}
        mul (32|M0)              acc0.0<1>:f   r36.0<1;1,0>:f    r2.5<0;1,0>:f    {Compacted}
        mul (32|M0)              r82.0<1>:f    r8.0<1;1,0>:f     r3.11<0;1,0>:f   {Compacted}
        mul (32|M0)              r78.0<1>:f    r8.0<1;1,0>:f     r3.0<0;1,0>:f    {Compacted}
        mul (32|M0)              r86.0<1>:f    r8.0<1;1,0>:f     r2.11<0;1,0>:f   {Compacted}
        mad (32|M0)              r68.0<1>:f    r74.0<1;0>:f      r68.0<1;0>:f      r2.4<0>:f        {Compacted}
        mul (32|M0)              r80.0<1>:f    r58.0<1;1,0>:f    r14.0<1;1,0>:f   {Compacted,F@7}
        mad (32|M0)              r22.0<1>:f    r12.0<1;0>:f      r22.0<1;0>:f      r2.4<0>:f        {Compacted,F@7}
        mad (32|M0)              r60.0<1>:f    r52.0<1;0>:f      r60.0<1;0>:f      r2.4<0>:f        {Compacted,F@7}
        mul (32|M0)              r76.0<1>:f    r30.0<1;1,0>:f    r14.0<1;1,0>:f   {Compacted,F@7}
        mad (32|M0)              r32.0<1>:f    acc2.0<1;0>:f     r32.0<1;0>:f      r2.4<0>:f        {Compacted}
        mad (32|M0)              r24.0<1>:f    acc0.0<1;0>:f     r24.0<1;0>:f      r2.4<0>:f        {Compacted}
        mul (32|M0)              r8.0<1>:f     r8.0<1;1,0>:f     r2.5<0;1,0>:f    {Compacted}
        mul (32|M0)              r12.0<1>:f    r22.0<1;1,0>:f    r14.0<1;1,0>:f   {Compacted,F@6}
        mad (32|M0)              acc2.0<1>:f   r84.0<1;0>:f      r68.0<1;0>:f      r40.0<1>:f       {Compacted}
        mad (32|M0)              r64.0<1>:f    r82.0<1;0>:f      r64.0<1;0>:f      r2.4<0>:f        {Compacted}
        mad (32|M0)              r56.0<1>:f    r78.0<1;0>:f      r56.0<1;0>:f      r2.4<0>:f        {Compacted}
        mad (32|M0)              acc0.0<1>:f   r80.0<1;0>:f      r60.0<1;0>:f      r40.0<1>:f       {Compacted,F@7}
        mul (32|M0)              r50.0<1>:f    r4.0<1;1,0>:f     r2.11<0;1,0>:f   {Compacted}
        mul (32|M0)              r36.0<1>:f    r4.0<1;1,0>:f     r2.5<0;1,0>:f    {Compacted}
        mad (32|M0)              r28.0<1>:f    r86.0<1;0>:f      r28.0<1;0>:f      r2.4<0>:f        {Compacted}
        mul (32|M0)              r74.0<1>:f    r4.0<1;1,0>:f     r3.11<0;1,0>:f   {Compacted}
        mul (32|M0)              r52.0<1>:f    r4.0<1;1,0>:f     r3.0<0;1,0>:f    {Compacted}
        mad (32|M0)              r20.0<1>:f    r8.0<1;0>:f       r20.0<1;0>:f      r2.4<0>:f        {Compacted,F@7}
        mad (32|M0)              r14.0<1>:f    r76.0<1;0>:f      r32.0<1;0>:f      r40.0<1>:f       {Compacted}
        mad (32|M0)              r12.0<1>:f    r12.0<1;0>:f      r24.0<1;0>:f      r40.0<1>:f       {Compacted}
        mad (32|M0)              r4.0<1>:f     acc0.0<1;0>:f     r56.0<1;0>:f      r10.0<1>:f       {Compacted,F@7}
        mad (32|M0)              r8.0<1>:f     acc2.0<1;0>:f     r64.0<1;0>:f      r10.0<1>:f       {Compacted}
        mad (32|M0)              acc2.0<1>:f   r14.0<1;0>:f      r28.0<1;0>:f      r10.0<1>:f       {Compacted,F@4}
        mad (32|M0)              r26.0<1>:f    r50.0<1;0>:f      r26.0<1;0>:f      r2.4<0>:f        {Compacted}
        mad (32|M0)              r62.0<1>:f    r74.0<1;0>:f      r62.0<1;0>:f      r2.4<0>:f        {Compacted}
        mad (32|M0)              r34.0<1>:f    r52.0<1;0>:f      r34.0<1;0>:f      r2.4<0>:f        {Compacted}
        mad (32|M0)              acc0.0<1>:f   r12.0<1;0>:f      r20.0<1;0>:f      r10.0<1>:f       {Compacted,F@7}
(W)     mov (32|M0)              r48.0<1>:ud   0x0:ud                             
(W)     mov (32|M0)              r46.0<1>:ud   0x0:ud                             
(W)     mov (32|M0)              r44.0<1>:ud   0x0:ud                             
        mad (32|M0)              r18.0<1>:f    r36.0<1;0>:f      r18.0<1;0>:f      r2.4<0>:f        {Compacted}
(W)     mov (32|M0)              r42.0<1>:ud   0x0:ud                             
        mad (32|M0)              r48.0<1>:f    r8.0<1;0>:f       r62.0<1;0>:f      r6.0<1>:f        {Compacted,A@4}
        mad (32|M0)              r46.0<1>:f    r4.0<1;0>:f       r34.0<1;0>:f      r6.0<1>:f        {Compacted,A@3}
        mad (32|M0)              r44.0<1>:f    acc2.0<1;0>:f     r26.0<1;0>:f      r6.0<1>:f        {Compacted,I@2}
        mad (32|M0)              r42.0<1>:f    acc0.0<1;0>:f     r18.0<1;0>:f      r6.0<1>:f        {Compacted,A@1}
(W)     add (16|M0)              r5.0<1>:f     r46.0<1;1,0>:f    r47.0<1;1,0>:f   {Compacted,F@3}
(W)     add (16|M0)              r4.0<1>:f     r44.0<1;1,0>:f    r45.0<1;1,0>:f   {Compacted,F@3}
(W)     add (16|M0)              r6.0<1>:f     r48.0<1;1,0>:f    r49.0<1;1,0>:f   {Compacted}
(W)     mov (8|M0)               r9.0<1>:ud    r5.8<1;1,0>:ud                   {Compacted,F@3}
(W)     mov (8|M0)               r10.0<1>:ud   r4.8<1;1,0>:ud                   {Compacted,F@2}
(W)     mov (8|M0)               r8.0<1>:ud    r6.8<1;1,0>:ud                   {Compacted,F@1}
(W)     add (8|M0)               r5.0<1>:f     r5.0<1;1,0>:f     r9.0<1;1,0>:f    {Compacted,I@3}
(W)     add (8|M0)               r4.0<1>:f     r4.0<1;1,0>:f     r10.0<1;1,0>:f   {Compacted,I@2}
(W)     add (8|M0)               r6.0<1>:f     r6.0<1;1,0>:f     r8.0<1;1,0>:f    {Compacted,I@1}
(W)     mov (4|M0)               r9.0<1>:ud    r5.4<1;1,0>:ud                   {Compacted,F@3}
(W)     mov (4|M0)               r10.0<1>:ud   r4.4<1;1,0>:ud                   {Compacted,F@2}
(W)     mov (4|M0)               r8.0<1>:ud    r6.4<1;1,0>:ud                   {Compacted,F@1}
(W)     add (4|M0)               r5.0<1>:f     r5.0<1;1,0>:f     r9.0<1;1,0>:f    {Compacted,I@3}
(W)     add (4|M0)               r4.0<1>:f     r4.0<1;1,0>:f     r10.0<1;1,0>:f   {Compacted,I@2}
(W)     add (4|M0)               r6.0<1>:f     r6.0<1;1,0>:f     r8.0<1;1,0>:f    {Compacted,I@1}
(W)     add (16|M0)              r7.0<1>:f     r42.0<1;1,0>:f    r43.0<1;1,0>:f   {Compacted}
(W)     add (1|M0)               r4.6<1>:f     r5.0<0;1,0>:f     r5.2<0;1,0>:f    {F@4}
(W)     add (1|M0)               r4.7<1>:f     r5.1<0;1,0>:f     r5.3<0;1,0>:f   
(W)     add (1|M0)               r4.4<1>:f     r4.0<0;1,0>:f     r4.2<0;1,0>:f    {Compacted,F@5}
(W)     add (1|M0)               r4.5<1>:f     r4.1<0;1,0>:f     r4.3<0;1,0>:f   
(W)     add (1|M0)               r4.8<1>:f     r6.0<0;1,0>:f     r6.2<0;1,0>:f    {F@6}
(W)     add (1|M0)               r4.9<1>:f     r6.1<0;1,0>:f     r6.3<0;1,0>:f   
(W)     add (1|M0)               r4.10<1>:f    r4.6<0;1,0>:f     r4.7<0;1,0>:f    {F@5}
(W)     add (1|M0)               r4.9<1>:f     r4.8<0;1,0>:f     r4.9<0;1,0>:f    {F@2}
(W)     add (1|M0)               r4.8<1>:f     r4.4<0;1,0>:f     r4.5<0;1,0>:f   
(W)     mov (8|M0)               r4.0<1>:ud    r7.8<1;1,0>:ud                   {Compacted,F@1}
(W)     add (8|M0)               r4.0<1>:f     r7.0<1;1,0>:f     r4.0<1;1,0>:f    {Compacted,I@1}
(W)     mov (4|M0)               r5.0<1>:ud    r4.4<1;1,0>:ud                   {Compacted,F@1}
(W)     add (4|M0)               r4.0<1>:f     r4.0<1;1,0>:f     r5.0<1;1,0>:f    {Compacted,I@1}
(W)     add (1|M0)               r4.4<1>:f     r4.0<0;1,0>:f     r4.2<0;1,0>:f    {Compacted,F@1}
(W)     add (1|M0)               r4.5<1>:f     r4.1<0;1,0>:f     r4.3<0;1,0>:f   
(W)     add (1|M0)               r4.2<1>:f     r4.4<0;1,0>:f     r4.5<0;1,0>:f    {F@1}
(~f3.0) goto (32|M0)                         L10544                  L10544                
L10408:
(W)     add (1|M0)               r8.0<1>:q     r1.7<0;1,0>:q     r1.0<0;1,0>:q    {Compacted}
(W)     mov (1|M0)               r4.2<1>:bf    r4.8<0;1,0>:f                   
(W)     mov (2|M0)               r4.16<1>:w    0x40:uv                              {F@1}
(W)     mov (1|M0)               r4.0<1>:bf    r4.9<0;1,0>:f                   
(W)     mov (1|M0)               r4.1<1>:bf    r4.10<0;1,0>:f                  
(W)     mov (1|M0)               r4.3<1>:bf    r4.2<0;1,0>:f                   
(W)     add (1|M0)               r6.0<1>:uq    r8.0<0;1,0>:uq    r4.16<0;1,0>:w   {I@1}
(W)     add (1|M0)               r6.1<1>:uq    r8.0<0;1,0>:uq    r4.17<0;1,0>:w  
(W)     send.ugm (2|M0)          null     r6  r4:1  0x0            0x04000584           {A@1,$10} // wr:2+1, rd:0; store.ugm.d32.a64
L10544:
        join (32|M0)                         L10560                                
L10560:
(W)     add3 (1|M0)              r1.13<1>:d    r2.14<0;0>:d      r1.12<0;0>:d      -r88.0<0>:d     
(W)     shl (1|M0)               r1.7<1>:q     r1.13<0;1,0>:d    2:w               {I@1}
(W)     add (1|M0)               r4.0<1>:q     r1.7<0;1,0>:q     r2.6<0;1,0>:q    {@1,$10.src}
(W)     send.ugm (1|M0)          r4       r4  null:0  0x0            0x02108580           {I@1,$11} // wr:1+0, rd:1; load.ugm.d32x1t.a64
(W)     cmp (32|M0)   (gt)f0.0   null<1>:d     r4.0<0;1,0>:d     0:w               {$11.dst}
(W&~f0.0) jmpi                               L13168                                
L10656:
(W)     mul (1|M0)               acc0.0<1>:ud  r1.10<0;1,0>:ud   r4.0<0;1,0>:uw  
(W)     macl (1|M0)              r6.0<1>:ud    r1.10<0;1,0>:ud   r4.0<0;1,0>:ud   {Compacted}
(W)     mul (1|M0)               acc0.0<1>:ud  r1.10<0;1,0>:ud   r4.0<0;1,0>:uw  
(W)     mach (1|M0)              r5.0<1>:d     r1.10<0;1,0>:ud   r4.0<0;1,0>:ud  
(W)     mul (1|M0)               acc0.0<1>:d   r4.0<0;1,0>:ud    r1.22<0;1,0>:uw 
(W)     macl (1|M0)              r4.0<1>:d     r4.0<0;1,0>:ud    r1.11<0;1,0>:d  
(W)     mul (1|M0)               acc0.0<1>:d   r3.6<0;1,0>:d     r3.4<0;1,0>:uw  
(W)     add (1|M0)               r6.1<1>:d     r5.0<0;1,0>:d     r4.0<0;1,0>:d    {Compacted,I@2}
(W)     macl (1|M0)              r4.0<1>:d     r3.6<0;1,0>:d     r3.2<0;1,0>:d    {Compacted}
(W)     shl (1|M0)               r1.7<1>:q     r6.0<0;1,0>:q     1:w               {I@2}
(W)     add (1|M0)               r54.2<1>:d    r2.6<0;1,0>:d     r4.0<0;1,0>:d    {I@2}
        mov (16|M0)              r12.0<1>:bf   r68.0<1;1,0>:f                  
        add (32|M0)              r4.0<1>:d     r16.0<1;1,0>:d    r54.2<0;1,0>:d   {Compacted,I@1}
        mov (16|M0)              r6.0<2>:ud    r4.0<1;1,0>:ud                   {Compacted,I@1}
        shl (16|M0)              r8.0<1>:q     r6.0<2;1,0>:d     1:w               {I@1}
        mov (16|M16)             r6.0<2>:ud    r5.0<1;1,0>:ud                   {Compacted}
        mov (16|M16)             r12.16<1>:bf  r69.0<1;1,0>:f                  
(W)     add (1|M0)               r1.7<1>:q     r1.7<0;1,0>:q     r2.4<0;1,0>:q   
        shl (16|M16)             r10.0<1>:q    r6.0<2;1,0>:d     1:w               {I@2}
        add (16|M0)              r4.0<1>:q     r1.7<0;1,0>:q     r8.0<1;1,0>:q    {Compacted,I@2}
        add (16|M16)             r6.0<1>:q     r1.7<0;1,0>:q     r10.0<1;1,0>:q   {Compacted,I@2}
        mov (32|M0)              r8.0<1>:ud    r12.0<1;1,0>:uw                  {F@1}
        send.ugm (32|M0)         null     r4  r8:2  0x0            0x08000B84           {I@1,$12} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r4.0<1>:d     r38.0<1;1,0>:d    r54.2<0;1,0>:d   {Compacted,$12.src}
        mov (16|M0)              r6.0<2>:ud    r4.0<1;1,0>:ud                   {Compacted,I@1}
        shl (16|M0)              r8.0<1>:q     r6.0<2;1,0>:d     1:w               {I@1}
        mov (16|M16)             r6.0<2>:ud    r5.0<1;1,0>:ud                   {Compacted}
        mov (16|M0)              r12.0<1>:bf   r66.0<1;1,0>:f                  
        mov (16|M16)             r12.16<1>:bf  r67.0<1;1,0>:f                  
        shl (16|M16)             r10.0<1>:q    r6.0<2;1,0>:d     1:w               {I@1}
        add (16|M0)              r4.0<1>:q     r1.7<0;1,0>:q     r8.0<1;1,0>:q    {Compacted}
        add (16|M16)             r6.0<1>:q     r1.7<0;1,0>:q     r10.0<1;1,0>:q   {Compacted,I@2}
        mov (32|M0)              r8.0<1>:ud    r12.0<1;1,0>:uw                  {F@1}
        send.ugm (32|M0)         null     r4  r8:2  0x0            0x08000B84           {I@1,$13} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r4.0<1>:d     r70.0<1;1,0>:d    r54.2<0;1,0>:d   {Compacted,$13.src}
        mov (16|M0)              r6.0<2>:ud    r4.0<1;1,0>:ud                   {Compacted,I@1}
        shl (16|M0)              r8.0<1>:q     r6.0<2;1,0>:d     1:w               {I@1}
        mov (16|M16)             r6.0<2>:ud    r5.0<1;1,0>:ud                   {Compacted}
        mov (16|M0)              r12.0<1>:bf   r64.0<1;1,0>:f                  
        mov (16|M16)             r12.16<1>:bf  r65.0<1;1,0>:f                  
        shl (16|M16)             r10.0<1>:q    r6.0<2;1,0>:d     1:w               {I@1}
        add (16|M0)              r4.0<1>:q     r1.7<0;1,0>:q     r8.0<1;1,0>:q    {Compacted}
        add (16|M16)             r6.0<1>:q     r1.7<0;1,0>:q     r10.0<1;1,0>:q   {Compacted,I@2}
        mov (32|M0)              r8.0<1>:ud    r12.0<1;1,0>:uw                  {F@1}
        send.ugm (32|M0)         null     r4  r8:2  0x0            0x08000B84           {I@1,$14} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r4.0<1>:d     r72.0<1;1,0>:d    r54.2<0;1,0>:d   {Compacted,$14.src}
        mov (16|M0)              r6.0<2>:ud    r4.0<1;1,0>:ud                   {Compacted,I@1}
        shl (16|M0)              r8.0<1>:q     r6.0<2;1,0>:d     1:w               {I@1}
        mov (16|M16)             r6.0<2>:ud    r5.0<1;1,0>:ud                   {Compacted}
        mov (16|M0)              r12.0<1>:bf   r62.0<1;1,0>:f                  
        mov (16|M16)             r12.16<1>:bf  r63.0<1;1,0>:f                  
        shl (16|M16)             r10.0<1>:q    r6.0<2;1,0>:d     1:w               {I@1}
(W)     or (1|M0)                r54.2<1>:d    r3.6<0;1,0>:d     1:w              
        add (16|M0)              r4.0<1>:q     r1.7<0;1,0>:q     r8.0<1;1,0>:q    {Compacted}
        add (16|M16)             r6.0<1>:q     r1.7<0;1,0>:q     r10.0<1;1,0>:q   {Compacted,I@3}
        mov (32|M0)              r8.0<1>:ud    r12.0<1;1,0>:uw                  {F@1}
        send.ugm (32|M0)         null     r4  r8:2  0x0            0x08000B84           {I@1,$0} // wr:4+2, rd:0; store.ugm.d16u32.a64
(W)     mul (1|M0)               acc0.0<1>:d   r54.2<0;1,0>:d    r3.4<0;1,0>:uw  
(W)     macl (1|M0)              r4.0<1>:d     r54.2<0;1,0>:d    r3.2<0;1,0>:d    {Compacted,$0.src}
        mov (16|M0)              r12.0<1>:bf   r60.0<1;1,0>:f                  
(W)     add (1|M0)               r54.2<1>:d    r2.6<0;1,0>:d     r4.0<0;1,0>:d    {I@1}
        mov (16|M16)             r12.16<1>:bf  r61.0<1;1,0>:f                  
        add (32|M0)              r4.0<1>:d     r16.0<1;1,0>:d    r54.2<0;1,0>:d   {Compacted,I@1}
        mov (16|M0)              r6.0<2>:ud    r4.0<1;1,0>:ud                   {Compacted,I@1}
        shl (16|M0)              r8.0<1>:q     r6.0<2;1,0>:d     1:w               {I@1}
        mov (16|M16)             r6.0<2>:ud    r5.0<1;1,0>:ud                   {Compacted}
        shl (16|M16)             r10.0<1>:q    r6.0<2;1,0>:d     1:w               {I@1}
        add (16|M0)              r4.0<1>:q     r1.7<0;1,0>:q     r8.0<1;1,0>:q    {Compacted}
        add (16|M16)             r6.0<1>:q     r1.7<0;1,0>:q     r10.0<1;1,0>:q   {Compacted,I@2}
        mov (32|M0)              r8.0<1>:ud    r12.0<1;1,0>:uw                  {F@1}
        send.ugm (32|M0)         null     r4  r8:2  0x0            0x08000B84           {I@1,$1} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r4.0<1>:d     r38.0<1;1,0>:d    r54.2<0;1,0>:d   {Compacted,$1.src}
        mov (16|M0)              r6.0<2>:ud    r4.0<1;1,0>:ud                   {Compacted,I@1}
        shl (16|M0)              r8.0<1>:q     r6.0<2;1,0>:d     1:w               {I@1}
        mov (16|M16)             r6.0<2>:ud    r5.0<1;1,0>:ud                   {Compacted}
        mov (16|M0)              r12.0<1>:bf   r58.0<1;1,0>:f                  
        mov (16|M16)             r12.16<1>:bf  r59.0<1;1,0>:f                  
        shl (16|M16)             r10.0<1>:q    r6.0<2;1,0>:d     1:w               {I@1}
        add (16|M0)              r4.0<1>:q     r1.7<0;1,0>:q     r8.0<1;1,0>:q    {Compacted}
        add (16|M16)             r6.0<1>:q     r1.7<0;1,0>:q     r10.0<1;1,0>:q   {Compacted,I@2}
        mov (32|M0)              r8.0<1>:ud    r12.0<1;1,0>:uw                  {F@1}
        send.ugm (32|M0)         null     r4  r8:2  0x0            0x08000B84           {I@1,$2} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r4.0<1>:d     r70.0<1;1,0>:d    r54.2<0;1,0>:d   {Compacted,$2.src}
        mov (16|M0)              r6.0<2>:ud    r4.0<1;1,0>:ud                   {Compacted,I@1}
        shl (16|M0)              r8.0<1>:q     r6.0<2;1,0>:d     1:w               {I@1}
        mov (16|M16)             r6.0<2>:ud    r5.0<1;1,0>:ud                   {Compacted}
        mov (16|M0)              r12.0<1>:bf   r56.0<1;1,0>:f                  
        mov (16|M16)             r12.16<1>:bf  r57.0<1;1,0>:f                  
        shl (16|M16)             r10.0<1>:q    r6.0<2;1,0>:d     1:w               {I@1}
        add (16|M0)              r4.0<1>:q     r1.7<0;1,0>:q     r8.0<1;1,0>:q    {Compacted}
        add (16|M16)             r6.0<1>:q     r1.7<0;1,0>:q     r10.0<1;1,0>:q   {Compacted,I@2}
        mov (32|M0)              r8.0<1>:ud    r12.0<1;1,0>:uw                  {F@1}
        send.ugm (32|M0)         null     r4  r8:2  0x0            0x08000B84           {I@1,$3} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r4.0<1>:d     r72.0<1;1,0>:d    r54.2<0;1,0>:d   {Compacted,$3.src}
        mov (16|M0)              r6.0<2>:ud    r4.0<1;1,0>:ud                   {Compacted,I@1}
        shl (16|M0)              r8.0<1>:q     r6.0<2;1,0>:d     1:w               {I@1}
        mov (16|M16)             r6.0<2>:ud    r5.0<1;1,0>:ud                   {Compacted}
        mov (16|M0)              r12.0<1>:bf   r34.0<1;1,0>:f                  
        mov (16|M16)             r12.16<1>:bf  r35.0<1;1,0>:f                  
        shl (16|M16)             r10.0<1>:q    r6.0<2;1,0>:d     1:w               {I@1}
(W)     or (1|M0)                r54.2<1>:d    r3.6<0;1,0>:d     2:w              
        add (16|M0)              r4.0<1>:q     r1.7<0;1,0>:q     r8.0<1;1,0>:q    {Compacted}
        add (16|M16)             r6.0<1>:q     r1.7<0;1,0>:q     r10.0<1;1,0>:q   {Compacted,I@3}
        mov (32|M0)              r8.0<1>:ud    r12.0<1;1,0>:uw                  {F@1}
        send.ugm (32|M0)         null     r4  r8:2  0x0            0x08000B84           {I@1,$4} // wr:4+2, rd:0; store.ugm.d16u32.a64
(W)     mul (1|M0)               acc0.0<1>:d   r54.2<0;1,0>:d    r3.4<0;1,0>:uw  
(W)     macl (1|M0)              r4.0<1>:d     r54.2<0;1,0>:d    r3.2<0;1,0>:d    {Compacted,$4.src}
        mov (16|M0)              r12.0<1>:bf   r32.0<1;1,0>:f                  
(W)     add (1|M0)               r54.2<1>:d    r2.6<0;1,0>:d     r4.0<0;1,0>:d    {I@1}
        mov (16|M16)             r12.16<1>:bf  r33.0<1;1,0>:f                  
        add (32|M0)              r4.0<1>:d     r16.0<1;1,0>:d    r54.2<0;1,0>:d   {Compacted,I@1}
        mov (16|M0)              r6.0<2>:ud    r4.0<1;1,0>:ud                   {Compacted,I@1}
        shl (16|M0)              r8.0<1>:q     r6.0<2;1,0>:d     1:w               {I@1}
        mov (16|M16)             r6.0<2>:ud    r5.0<1;1,0>:ud                   {Compacted}
        shl (16|M16)             r10.0<1>:q    r6.0<2;1,0>:d     1:w               {I@1}
        add (16|M0)              r4.0<1>:q     r1.7<0;1,0>:q     r8.0<1;1,0>:q    {Compacted}
        add (16|M16)             r6.0<1>:q     r1.7<0;1,0>:q     r10.0<1;1,0>:q   {Compacted,I@2}
        mov (32|M0)              r8.0<1>:ud    r12.0<1;1,0>:uw                  {F@1}
        send.ugm (32|M0)         null     r4  r8:2  0x0            0x08000B84           {I@1,$5} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r4.0<1>:d     r38.0<1;1,0>:d    r54.2<0;1,0>:d   {Compacted,$5.src}
        mov (16|M0)              r6.0<2>:ud    r4.0<1;1,0>:ud                   {Compacted,I@1}
        shl (16|M0)              r8.0<1>:q     r6.0<2;1,0>:d     1:w               {I@1}
        mov (16|M16)             r6.0<2>:ud    r5.0<1;1,0>:ud                   {Compacted}
        mov (16|M0)              r12.0<1>:bf   r30.0<1;1,0>:f                  
        mov (16|M16)             r12.16<1>:bf  r31.0<1;1,0>:f                  
        shl (16|M16)             r10.0<1>:q    r6.0<2;1,0>:d     1:w               {I@1}
        add (16|M0)              r4.0<1>:q     r1.7<0;1,0>:q     r8.0<1;1,0>:q    {Compacted}
        add (16|M16)             r6.0<1>:q     r1.7<0;1,0>:q     r10.0<1;1,0>:q   {Compacted,I@2}
        mov (32|M0)              r8.0<1>:ud    r12.0<1;1,0>:uw                  {F@1}
        send.ugm (32|M0)         null     r4  r8:2  0x0            0x08000B84           {I@1,$6} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r4.0<1>:d     r70.0<1;1,0>:d    r54.2<0;1,0>:d   {Compacted,$6.src}
        mov (16|M0)              r6.0<2>:ud    r4.0<1;1,0>:ud                   {Compacted,I@1}
        shl (16|M0)              r8.0<1>:q     r6.0<2;1,0>:d     1:w               {I@1}
        mov (16|M16)             r6.0<2>:ud    r5.0<1;1,0>:ud                   {Compacted}
        mov (16|M0)              r12.0<1>:bf   r28.0<1;1,0>:f                  
        mov (16|M16)             r12.16<1>:bf  r29.0<1;1,0>:f                  
        shl (16|M16)             r10.0<1>:q    r6.0<2;1,0>:d     1:w               {I@1}
        add (16|M0)              r4.0<1>:q     r1.7<0;1,0>:q     r8.0<1;1,0>:q    {Compacted}
        add (16|M16)             r6.0<1>:q     r1.7<0;1,0>:q     r10.0<1;1,0>:q   {Compacted,I@2}
        mov (32|M0)              r8.0<1>:ud    r12.0<1;1,0>:uw                  {F@1}
        send.ugm (32|M0)         null     r4  r8:2  0x0            0x08000B84           {I@1,$7} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r4.0<1>:d     r72.0<1;1,0>:d    r54.2<0;1,0>:d   {Compacted,$7.src}
        mov (16|M0)              r6.0<2>:ud    r4.0<1;1,0>:ud                   {Compacted,I@1}
        shl (16|M0)              r8.0<1>:q     r6.0<2;1,0>:d     1:w               {I@1}
        mov (16|M16)             r6.0<2>:ud    r5.0<1;1,0>:ud                   {Compacted}
        mov (16|M0)              r12.0<1>:bf   r26.0<1;1,0>:f                  
        mov (16|M16)             r12.16<1>:bf  r27.0<1;1,0>:f                  
        shl (16|M16)             r10.0<1>:q    r6.0<2;1,0>:d     1:w               {I@1}
(W)     or (1|M0)                r54.2<1>:d    r3.6<0;1,0>:d     3:w              
        add (16|M0)              r4.0<1>:q     r1.7<0;1,0>:q     r8.0<1;1,0>:q    {Compacted}
        add (16|M16)             r6.0<1>:q     r1.7<0;1,0>:q     r10.0<1;1,0>:q   {Compacted,I@3}
        mov (32|M0)              r8.0<1>:ud    r12.0<1;1,0>:uw                  {F@1}
        send.ugm (32|M0)         null     r4  r8:2  0x0            0x08000B84           {I@1,$8} // wr:4+2, rd:0; store.ugm.d16u32.a64
(W)     mul (1|M0)               acc0.0<1>:d   r54.2<0;1,0>:d    r3.4<0;1,0>:uw  
(W)     macl (1|M0)              r4.0<1>:d     r54.2<0;1,0>:d    r3.2<0;1,0>:d    {Compacted,$8.src}
        mov (16|M0)              r12.0<1>:bf   r24.0<1;1,0>:f                  
(W)     add (1|M0)               r54.2<1>:d    r2.6<0;1,0>:d     r4.0<0;1,0>:d    {I@1}
        mov (16|M16)             r12.16<1>:bf  r25.0<1;1,0>:f                  
        add (32|M0)              r4.0<1>:d     r16.0<1;1,0>:d    r54.2<0;1,0>:d   {Compacted,I@1}
        mov (16|M0)              r6.0<2>:ud    r4.0<1;1,0>:ud                   {Compacted,I@1}
        shl (16|M0)              r8.0<1>:q     r6.0<2;1,0>:d     1:w               {I@1}
        mov (16|M16)             r6.0<2>:ud    r5.0<1;1,0>:ud                   {Compacted}
        shl (16|M16)             r10.0<1>:q    r6.0<2;1,0>:d     1:w               {I@1}
        add (16|M0)              r4.0<1>:q     r1.7<0;1,0>:q     r8.0<1;1,0>:q    {Compacted}
        add (16|M16)             r6.0<1>:q     r1.7<0;1,0>:q     r10.0<1;1,0>:q   {Compacted,I@2}
        mov (32|M0)              r8.0<1>:ud    r12.0<1;1,0>:uw                  {F@1}
        send.ugm (32|M0)         null     r4  r8:2  0x0            0x08000B84           {I@1,$9} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r4.0<1>:d     r38.0<1;1,0>:d    r54.2<0;1,0>:d   {Compacted,$9.src}
        mov (16|M0)              r6.0<2>:ud    r4.0<1;1,0>:ud                   {Compacted,I@1}
        shl (16|M0)              r8.0<1>:q     r6.0<2;1,0>:d     1:w               {I@1}
        mov (16|M16)             r6.0<2>:ud    r5.0<1;1,0>:ud                   {Compacted}
        mov (16|M0)              r12.0<1>:bf   r22.0<1;1,0>:f                  
        mov (16|M16)             r12.16<1>:bf  r23.0<1;1,0>:f                  
        shl (16|M16)             r10.0<1>:q    r6.0<2;1,0>:d     1:w               {I@1}
        add (16|M0)              r4.0<1>:q     r1.7<0;1,0>:q     r8.0<1;1,0>:q    {Compacted}
        add (16|M16)             r6.0<1>:q     r1.7<0;1,0>:q     r10.0<1;1,0>:q   {Compacted,I@2}
        mov (32|M0)              r8.0<1>:ud    r12.0<1;1,0>:uw                  {F@1}
        send.ugm (32|M0)         null     r4  r8:2  0x0            0x08000B84           {I@1,$10} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r4.0<1>:d     r70.0<1;1,0>:d    r54.2<0;1,0>:d   {Compacted,$10.src}
        mov (16|M0)              r6.0<2>:ud    r4.0<1;1,0>:ud                   {Compacted,I@1}
        shl (16|M0)              r8.0<1>:q     r6.0<2;1,0>:d     1:w               {I@1}
        mov (16|M16)             r6.0<2>:ud    r5.0<1;1,0>:ud                   {Compacted}
        mov (16|M0)              r12.0<1>:bf   r20.0<1;1,0>:f                  
        mov (16|M16)             r12.16<1>:bf  r21.0<1;1,0>:f                  
        shl (16|M16)             r10.0<1>:q    r6.0<2;1,0>:d     1:w               {I@1}
        add (16|M0)              r4.0<1>:q     r1.7<0;1,0>:q     r8.0<1;1,0>:q    {Compacted}
        add (16|M16)             r6.0<1>:q     r1.7<0;1,0>:q     r10.0<1;1,0>:q   {Compacted,I@2}
        mov (32|M0)              r8.0<1>:ud    r12.0<1;1,0>:uw                  {F@1}
        send.ugm (32|M0)         null     r4  r8:2  0x0            0x08000B84           {I@1,$11} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r4.0<1>:d     r72.0<1;1,0>:d    r54.2<0;1,0>:d   {Compacted,$11.src}
        mov (16|M0)              r6.0<2>:ud    r4.0<1;1,0>:ud                   {Compacted,I@1}
        shl (16|M0)              r8.0<1>:q     r6.0<2;1,0>:d     1:w               {I@1}
        mov (16|M16)             r6.0<2>:ud    r5.0<1;1,0>:ud                   {Compacted}
        mov (16|M0)              r12.0<1>:bf   r18.0<1;1,0>:f                  
        mov (16|M16)             r12.16<1>:bf  r19.0<1;1,0>:f                  
        shl (16|M16)             r10.0<1>:q    r6.0<2;1,0>:d     1:w               {I@1}
        add (16|M0)              r4.0<1>:q     r1.7<0;1,0>:q     r8.0<1;1,0>:q    {Compacted}
        add (16|M16)             r6.0<1>:q     r1.7<0;1,0>:q     r10.0<1;1,0>:q   {Compacted,I@2}
        mov (32|M0)              r8.0<1>:ud    r12.0<1;1,0>:uw                  {F@1}
        send.ugm (32|M0)         null     r4  r8:2  0x0            0x08000B84           {I@1,$15} // wr:4+2, rd:0; store.ugm.d16u32.a64
L13168:
(W)     add (1|M0)               r1.13<1>:d    r1.12<0;1,0>:d    1:w              
(W)     cmp (32|M0)   (lt)f2.0   null<1>:d     r1.13<0;1,0>:d    r88.1<0;1,0>:d   {I@1}
(W&~f2.0) jmpi                               L13248                                
L13216:
(W)     mov (1|M0)               r1.12<1>:d    r1.13<0;1,0>:d                  
(W)     jmpi                                 L4952                                
L13248:
(W)     mov (16|M0)              r112.0<1>:f   r55.0<1;1,0>:f                   {Compacted}
(W)     send.gtwy (1|M0)         null     r112  null:0  0x0            0x02000010           {EOT,F@1,$12} // wr:1+0, rd:0; end of thread
L13272:
(W)     mov (16|M0)              null<1>:ud    0xBA0B4088:ud                             
(W)     mov (16|M0)              null<1>:ud    0xAC922C90:ud                             
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
        illegal                
        illegal                
