L0:
(W)     and (1|M0)               r127.0<1>:ud  r0.0<0;1,0>:ud    0xFFFFFFC0:ud             
(W)     add (1|M0)               r127.0<1>:ud  r127.0<0;1,0>:ud  0x0:ud              {I@1}
(W)     send.ugm (1|M0)          r2       r127  null:0  0xFF000000            0x6219D500           {A@1,$0} // wr:1+0, rd:1; load.ugm.d32x16t.a32.ca.cc.bti[255]
(W)     send.ugm (1|M0)          r3       r127  null:0  0xFF040000            0x6219C500           {$1} // wr:1+0, rd:1; load.ugm.d32x8t.a32.ca.cc.bti[255][A+0x40]
(W)     mov (16|M0)              r13.0<1>:ud   r0.0<1;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               r4.0<1>:f     9.18355e-41:f                              
(W)     and (1|M0)               r1.9<1>:ud    r13.5<0;1,0>:ud   0xFFFFFC00:ud              {I@1}
(W)     or (1|M0)                cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x400004C0:ud              {A@1}
(W)     mov (8|M0)               r56.0<1>:w    0x76543210:v                               {A@1}
(W)     cmp (32|M0)   (eq)f0.0   null<1>:d     r3.1<0;1,0>:d     0:w               {$1.dst}
(W)     add (8|M0)               r56.8<1>:w    r56.0<1;1,0>:w    8:w               {I@2}
(W)     mov (1|M0)               r3.14<1>:d    r13.1<0;1,0>:d                  
(W)     add (16|M0)              r56.16<1>:w   r56.0<1;1,0>:w    16:w               {I@2}
(W)     mov (1|M0)               r3.60<2>:b    r13.8<0;1,0>:b                  
(W&~f0.0) jmpi                               L264                                
L232:
(W)     mov (1|M0)               r3.9<1>:d     -1:w                              
(W)     jmpi                                 L824                                
L264:
(W)     asr (1|M0)               r1.15<1>:d    r3.1<0;1,0>:d     31:w              
(W)     asr (1|M0)               r1.14<1>:d    r3.3<0;1,0>:d     31:w              
(W)     add (1|M0)               r1.10<1>:d    r1.15<0;1,0>:d    r3.1<0;1,0>:d    {I@2}
(W)     xor (1|M0)               r1.11<1>:d    r1.10<0;1,0>:d    r1.15<0;1,0>:d   {I@1}
(W)     add (1|M0)               r1.10<1>:d    r1.14<0;1,0>:d    r3.3<0;1,0>:d   
(W)     xor (1|M0)               r3.7<1>:d     r1.10<0;1,0>:d    r1.14<0;1,0>:d   {I@1}
(W)     xor (1|M0)               cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x30:uw              {A@1}
(W)     mov (1|M0)               r3.6<1>:f     r1.11<0;1,0>:ud                  {A@1}
(W)     mov (1|M0)               r3.5<1>:f     r3.7<0;1,0>:ud                   {I@2}
(W)     mov (1|M0)               r1.10<1>:ud   r3.6<0;1,0>:f                    {F@2}
(W)     math.inv (1|M0)          r3.8<1>:f     r3.6<0;1,0>:f                   
(W)     add (1|M0)               r3.12<1>:d    r1.11<0;1,0>:d    -r1.10<0;1,0>:d  {I@1}
(W)     mov (1|M0)               r1.10<1>:f    0xB4C00000:f                               {I@1}
(W)     mov (1|M0)               r1.12<1>:f    r3.12<0;1,0>:ud                 
(W)     mad (1|M0)               r3.11<1>:f    r3.8<0;0>:f       r1.10<0;0>:f      r3.8<0>:f        {A@1}
(W)     mov (1|M0)               r1.10<1>:ud   r3.5<0;1,0>:f                    {F@1}
(W)     mul (1|M0)               r3.8<1>:f     r3.5<0;1,0>:f     r3.11<0;1,0>:f  
(W)     add (1|M0)               r3.13<1>:d    r3.7<0;1,0>:d     -r1.10<0;1,0>:d  {I@1}
(W)     mov (1|M0)               r3.10<1>:ud   r3.8<0;1,0>:f                    {F@1}
(W)     mov (1|M0)               r1.13<1>:f    r3.13<0;1,0>:ud                  {I@2}
(W)     mov (1|M0)               r3.8<1>:f     r3.10<0;1,0>:ud                  {I@1}
(W)     mad (1|M0)               r3.5<1>:f     r3.5<0;0>:f       r3.8<0;0>:f       -r3.6<0>:f       {F@1}
(W)     mad (1|M0)               r1.10<1>:f    r1.13<0;0>:f      r3.8<0;0>:f       -r1.12<0>:f     
(W)     add (1|M0)               r1.10<1>:f    r3.5<0;1,0>:f     r1.10<0;1,0>:f   {F@1}
(W)     mul (1|M0)               r3.5<1>:f     r3.11<0;1,0>:f    r1.10<0;1,0>:f   {F@1}
(W)     xor (1|M0)               cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x30:uw              {A@1}
(W)     mov (1|M0)               r1.10<1>:ud   r3.5<0;1,0>:f                    {A@1}
(W)     xor (1|M0)               r3.6<1>:d     r1.15<0;1,0>:d    r1.14<0;1,0>:d  
(W)     add (1|M0)               r3.5<1>:d     r1.10<0;1,0>:d    r3.10<0;1,0>:d   {I@2}
(W)     mul (1|M0)               acc0.0<1>:d   r3.5<0;1,0>:d     r1.22<0;1,0>:uw  {I@1}
(W)     macl (1|M0)              r5.0<1>:d     r3.5<0;1,0>:d     r1.11<0;1,0>:d  
(W)     add (1|M0)               r1.10<1>:d    r3.7<0;1,0>:d     -r5.0<0;1,0>:d   {I@1}
(W)     cmp (1|M0)    (ge)f2.0   r1.10<1>:ud   r1.10<0;1,0>:ud   r1.11<0;1,0>:ud  {I@1}
(W)     add3 (1|M0)              r1.10<1>:d    r3.5<0;0>:d       r3.6<0;0>:d       -r1.10<0>:d      {I@1}
(W)     bfn.(s0^s1^s2) (1|M0)    r3.9<1>:ud    r1.10<0;0>:ud     r1.15<0;0>:ud     r1.14<0>:ud      {I@1}
L824:
(W)     mov (1|M0)               r1.10<1>:d    r3.60<0;1,0>:ub                 
(W)     shl (1|M0)               r1.11<1>:d    r3.14<0;1,0>:d    5:w              
(W)     shl (1|M0)               r1.10<1>:d    r1.10<0;1,0>:d    2:w               {I@2}
(W)     add (1|M0)               r3.5<1>:d     r1.11<0;1,0>:d    r1.10<0;1,0>:d   {I@1}
(W)     cmp (32|M0)   (lt)f3.0   null<1>:d     r3.5<0;1,0>:d     r3.4<0;1,0>:d    {I@1}
(W&~f3.0) jmpi                               L13384                                
L920:
(W)     mov (1|M0)               r3.6<1>:d     r13.6<0;1,0>:d                  
(W)     mov (1|M0)               r3.5<1>:q     r13.7<0;1,0>:ud                 
(W)     shl (1|M0)               r1.6<1>:q     r3.6<0;1,0>:ud    2:w               {I@2}
(W)     shl (1|M0)               r1.7<1>:q     r3.6<0;1,0>:ud    1:w              
(W)     add (1|M0)               r6.0<1>:q     r1.6<0;1,0>:q     r2.2<0;1,0>:q    {Compacted,@2,$0.dst}
(W)     add (1|M0)               r10.0<1>:q    r1.7<0;1,0>:q     r2.3<0;1,0>:q    {Compacted,I@2}
(W)     send.ugm (1|M0)          r6       r6  null:0  0x0            0x02108580           {I@2,$2} // wr:1+0, rd:1; load.ugm.d32x1t.a64
(W)     send.ugm (1|M0)          r5       r10  null:0  0x0            0x04100B80           {I@1,$3} // wr:2+0, rd:1; load.ugm.d16u32.a64
(W)     shl (1|M0)               r3.6<1>:q     r3.5<0;1,0>:q     2:w              
(W)     mov (1|M0)               r1.5<1>:uq    0x0:uw                             
(W)     add (1|M0)               r8.0<1>:q     r3.6<0;1,0>:q     r2.6<0;1,0>:q    {I@2}
(W)     cmp (32|M0)   (eq)f2.0   null<1>:d     r2.14<0;1,0>:d    r1.10<0;1,0>:d   {I@2}
(W)     send.ugm (1|M0)          r7       r8  null:0  0x0            0x02108580           {I@2,$4} // wr:1+0, rd:1; load.ugm.d32x1t.a64
(W&f2.0) cmp (32|M0)  (eq)f2.0   null<1>:d     r2.15<0;1,0>:d    r1.11<0;1,0>:d  
        shl (32|M0)              r14.0<1>:d    r56.0<1;1,0>:uw   2:w              
(W)     mov (1|M0)               r3.8<1>:ud    0xBF317200:ud                             
(W)     mov (1|M0)               r3.7<1>:ud    0xB5BFBE8E:ud                             
        sync.nop                             null                             {Compacted,I@5}
(W)     mul (1|M0)               r1.10<1>:f    r6.0<0;1,0>:f     1.442695e+00:f               {$2.dst}
(W)     shl (1|M0)               r4.3<1>:ud    r5.0<0;1,0>:uw    0x10:uw              {$3.dst}
(W)     rndz (1|M0)              r5.0<1>:f     r1.10<0;1,0>:f                   {Compacted,A@1}
(W)     mov (1|M0)               r1.10<1>:ud   0xBF317200:ud                              {F@1}
(W)     cmp (1|M0)    (lt)f0.0   null<1>:f     r6.0<0;1,0>:f     -105.0:f              
(W)     mad (1|M0)               r1.11<1>:f    r6.0<0;0>:f       r1.10<0;0>:f      r5.0<0>:f        {I@1}
(W)     mov (1|M0)               r1.10<1>:ud   0xB5BFBE8E:ud                              {F@1}
(W)     cmp (1|M0)    (gt)f3.0   null<1>:f     r6.0<0;1,0>:f     105.0:f              
(W)     mad (1|M0)               r1.10<1>:f    r1.11<0;0>:f      r1.10<0;0>:f      r5.0<0>:f        {I@1}
(W)     math.exp (1|M0)          r6.0<1>:f     r5.0<0;1,0>:f                    {F@2}
(W)     mul (1|M0)               r5.1<1>:f     r1.10<0;1,0>:f    1.442695e+00:f               {F@1}
(W)     asr (1|M0)               r3.12<1>:d    r7.0<0;1,0>:d     31:w               {$4.dst}
(W)     math.exp (1|M0)          r6.1<1>:f     r5.1<0;1,0>:f                    {F@1}
(W)     mul (1|M0)               r1.10<1>:f    r6.0<0;1,0>:f     r6.1<0;1,0>:f    {M@1}
(W&~f0.0) sel (1|M0)             r1.10<1>:f    r1.10<0;1,0>:f    0.0:f               {F@1}
(W&~f3.0) sel (1|M0)             r4.2<1>:f     r1.10<0;1,0>:f    inf:f               {F@1}
(W&f2.0) jmpi                                L1912                                
L1448:
(W)     add (1|M0)               r6.0<1>:q     r2.7<0;1,0>:q     r3.5<0;1,0>:q    {Compacted}
(W)     asr (1|M0)               r1.10<1>:d    r3.0<0;1,0>:d     31:w               {F@1}
(W)     send.ugm (1|M0)          r5       r6  null:0  0x0            0x04100980           {I@2,$5} // wr:2+0, rd:1; load.ugm.d8u32.a64
(W)     mul (1|M0)               acc0.0<1>:ud  r7.0<0;1,0>:ud    r3.0<0;1,0>:uw  
(W)     mov (1|M0)               r9.0<2>:b     r5.0<0;1,0>:b                    {$5.dst}
(W)     macl (1|M0)              r5.0<1>:ud    r7.0<0;1,0>:ud    r3.0<0;1,0>:ud   {Compacted}
(W)     mul (1|M0)               acc0.0<1>:ud  r7.0<0;1,0>:ud    r3.0<0;1,0>:uw  
(W)     mach (1|M0)              r8.0<1>:d     r7.0<0;1,0>:ud    r3.0<0;1,0>:ud  
(W)     mul (1|M0)               acc0.0<1>:d   r7.0<0;1,0>:ud    r1.20<0;1,0>:uw  {I@6}
(W)     macl (1|M0)              r6.0<1>:d     r7.0<0;1,0>:ud    r1.10<0;1,0>:d  
(W)     mul (1|M0)               acc0.0<1>:d   r3.0<0;1,0>:ud    r3.24<0;1,0>:uw 
(W)     add (1|M0)               r8.0<1>:d     r8.0<0;1,0>:d     r6.0<0;1,0>:d    {Compacted,I@2}
(W)     macl (1|M0)              r6.0<1>:d     r3.0<0;1,0>:ud    r3.12<0;1,0>:d  
(W)     add (1|M0)               r5.1<1>:d     r8.0<0;1,0>:d     r6.0<0;1,0>:d    {Compacted,I@1}
(W)     shl (1|M0)               r1.5<1>:q     r5.0<0;1,0>:q     1:w               {I@1}
(W)     add (1|M0)               r2.3<1>:q     r1.5<0;1,0>:q     r2.4<0;1,0>:q    {I@1}
(W)     mov (1|M0)               r1.20<1>:w    r9.0<0;1,0>:ub                  
(W)     cmp (32|M0)   (eq)f2.0   null<2>:w     r1.20<0;1,0>:w    0:w               {I@1}
(W&~f2.0) jmpi                               L1880                                
L1720:
        mov (32|M0)              r32.0<1>:ud   0x0:ud                              {Compacted}
        mov (32|M0)              r30.0<1>:ud   0x0:ud                              {Compacted}
        mov (32|M0)              r28.0<1>:ud   0x0:ud                              {Compacted}
        mov (32|M0)              r26.0<1>:ud   0x0:ud                              {Compacted}
        mov (32|M0)              r24.0<1>:ud   0x0:ud                              {Compacted}
        mov (32|M0)              r22.0<1>:ud   0x0:ud                              {Compacted}
        mov (32|M0)              r20.0<1>:ud   0x0:ud                              {Compacted}
        mov (32|M0)              r50.0<1>:ud   0x0:ud                              {Compacted}
        mov (32|M0)              r52.0<1>:f    0.0:f                               {Compacted}
        mov (32|M0)              r54.0<1>:f    0.0:f                               {Compacted}
        mov (32|M0)              r58.0<1>:f    0.0:f                               {Compacted}
        mov (32|M0)              r62.0<1>:f    0.0:f                               {Compacted}
        mov (32|M0)              r66.0<1>:f    0.0:f                               {Compacted}
        mov (32|M0)              r68.0<1>:f    0.0:f                               {Compacted}
        mov (32|M0)              r70.0<1>:f    0.0:f                               {Compacted}
        mov (32|M0)              r72.0<1>:f    0.0:f                               {Compacted}
(W)     mov (1|M0)               r1.7<1>:uq    r2.3<0;1,0>:uq                  
(W)     jmpi                                 L4928                                
L1880:
(W)     mov (1|M0)               r1.5<1>:uq    r2.3<0;1,0>:uq                  
(W)     jmpi                                 L2096                                
L1912:
(W)     asr (1|M0)               r2.4<1>:d     r3.0<0;1,0>:d     31:w               {Compacted}
(W)     mul (1|M0)               acc0.0<1>:ud  r7.0<0;1,0>:ud    r3.0<0;1,0>:uw  
(W)     macl (1|M0)              r5.0<1>:ud    r7.0<0;1,0>:ud    r3.0<0;1,0>:ud   {Compacted}
(W)     mul (1|M0)               acc0.0<1>:ud  r7.0<0;1,0>:ud    r3.0<0;1,0>:uw  
(W)     mach (1|M0)              r8.0<1>:d     r7.0<0;1,0>:ud    r3.0<0;1,0>:ud  
(W)     mul (1|M0)               acc0.0<1>:d   r7.0<0;1,0>:ud    r2.8<0;1,0>:uw   {I@5}
(W)     macl (1|M0)              r6.0<1>:d     r7.0<0;1,0>:ud    r2.4<0;1,0>:d   
(W)     mul (1|M0)               acc0.0<1>:d   r3.0<0;1,0>:ud    r3.24<0;1,0>:uw 
(W)     add (1|M0)               r8.0<1>:d     r8.0<0;1,0>:d     r6.0<0;1,0>:d    {Compacted,I@2}
(W)     macl (1|M0)              r6.0<1>:d     r3.0<0;1,0>:ud    r3.12<0;1,0>:d  
(W)     add (1|M0)               r5.1<1>:d     r8.0<0;1,0>:d     r6.0<0;1,0>:d    {Compacted,I@1}
(W)     shl (1|M0)               r2.2<1>:q     r5.0<0;1,0>:q     1:w               {Compacted,I@1}
(W)     add (1|M0)               r2.3<1>:q     r2.2<0;1,0>:q     r2.4<0;1,0>:q    {I@1}
(W)     mov (1|M0)               r1.5<1>:uq    r2.3<0;1,0>:uq                   {I@1}
L2096:
(W)     mul (1|M0)               acc0.0<1>:d   r3.6<0;1,0>:d     r3.4<0;1,0>:uw  
(W)     macl (1|M0)              r3.0<1>:d     r3.6<0;1,0>:d     r3.2<0;1,0>:d    {Compacted}
(W)     or (1|M0)                r2.4<1>:d     r3.5<0;1,0>:d     1:w              
(W)     mul (1|M0)               acc0.0<1>:d   r3.0<0;1,0>:d     r3.8<0;1,0>:uw   {I@2}
(W)     macl (1|M0)              r3.0<1>:d     r3.0<0;1,0>:d     r3.4<0;1,0>:d    {Compacted}
(W)     mul (1|M0)               acc0.0<1>:d   r3.5<0;1,0>:d     r3.4<0;1,0>:uw  
(W)     or (1|M0)                r2.5<1>:d     r3.5<0;1,0>:d     2:w              
(W)     macl (1|M0)              r5.0<1>:d     r3.5<0;1,0>:d     r3.2<0;1,0>:d   
(W)     mul (1|M0)               acc0.0<1>:d   r2.4<0;1,0>:d     r3.4<0;1,0>:uw   {I@6}
(W)     macl (1|M0)              r6.0<1>:d     r2.4<0;1,0>:d     r3.2<0;1,0>:d    {Compacted}
(W)     mul (1|M0)               acc0.0<1>:d   r2.5<0;1,0>:d     r3.4<0;1,0>:uw   {I@4}
(W)     add (1|M0)               r2.4<1>:d     r3.0<0;1,0>:d     r5.0<0;1,0>:d    {Compacted,I@4}
(W)     macl (1|M0)              r5.0<1>:d     r2.5<0;1,0>:d     r3.2<0;1,0>:d   
(W)     add (1|M0)               r2.5<1>:d     r3.0<0;1,0>:d     r6.0<0;1,0>:d    {I@4}
(W)     mov (1|M0)               r1.8<1>:ud    a0.2<0;1,0>:ud                  
        add (32|M0)              r38.0<1>:d    r14.0<1;1,0>:d    r2.5<0;1,0>:d    {Compacted,I@2}
(W)     shr (1|M0)               a0.2<1>:ud    r1.9<0;1,0>:ud    0x4:ud              {F@1}
        mov (16|M16)             r74.0<2>:ud   r39.0<1;1,0>:ud                  {Compacted,I@2}
        shl (16|M16)             r116.0<1>:q   r74.0<2;1,0>:d    1:w               {I@1}
(W)     send.ugm (1|M0)          null     r4  r116:2  0x83000000:a0.2        0x4200E504           {I@1,$6} // wr:1+2, rd:0; store.ugm.d32x32t.a32.ss[a0.2][A-0xFA00]
(W)     or (1|M0)                r2.6<1>:d     r3.5<0;1,0>:d     3:w              
        add (32|M0)              r44.0<1>:d    r14.0<1;1,0>:d    r2.4<0;1,0>:d    {Compacted}
        add3 (32|M0)             r42.0<1>:d    r14.0<1;0>:d      r2.4<0;0>:d       1:w              
        add3 (32|M0)             r16.0<1>:d    r14.0<1;0>:d      r2.4<0;0>:d       2:w              
        add3 (32|M0)             r40.0<1>:d    r14.0<1;0>:d      r2.4<0;0>:d       3:w              
        add3 (32|M0)             r36.0<1>:d    r14.0<1;0>:d      r2.5<0;0>:d       1:w              
        add3 (32|M0)             r32.0<1>:d    r14.0<1;0>:d      r2.5<0;0>:d       2:w              
(W)     mul (1|M0)               acc0.0<1>:d   r2.6<0;1,0>:d     r3.4<0;1,0>:uw   {I@7}
        add3 (32|M0)             r30.0<1>:d    r14.0<1;0>:d      r2.5<0;0>:d       3:w              
(W)     macl (1|M0)              r6.0<1>:d     r2.6<0;1,0>:d     r3.2<0;1,0>:d    {Compacted}
(W)     add (1|M0)               r2.6<1>:d     r3.0<0;1,0>:d     r5.0<0;1,0>:d   
        mov (16|M0)              r34.0<2>:ud   r44.0<1;1,0>:ud                  {Compacted,I@7}
        mov (16|M16)             r18.0<2>:ud   r45.0<1;1,0>:ud                  {Compacted}
        mov (16|M0)              r46.0<2>:ud   r42.0<1;1,0>:ud                  {Compacted,I@7}
        mov (16|M0)              r48.0<2>:ud   r16.0<1;1,0>:ud                  {Compacted,I@7}
        mov (16|M16)             r60.0<2>:ud   r41.0<1;1,0>:ud                  {Compacted,I@7}
        mov (16|M16)             r64.0<2>:ud   r37.0<1;1,0>:ud                  {Compacted,I@7}
        mov (16|M16)             r44.0<2>:ud   r43.0<1;1,0>:ud                  {Compacted}
        mov (16|M16)             r42.0<2>:ud   r17.0<1;1,0>:ud                  {Compacted}
        mov (16|M0)              r16.0<2>:ud   r40.0<1;1,0>:ud                  {Compacted}
        mov (16|M0)              r40.0<2>:ud   r38.0<1;1,0>:ud                  {Compacted}
        mov (16|M0)              r38.0<2>:ud   r36.0<1;1,0>:ud                  {Compacted}
        mov (16|M0)              r36.0<2>:ud   r32.0<1;1,0>:ud                  {Compacted}
(W)     add (1|M0)               r2.7<1>:d     r3.0<0;1,0>:d     r6.0<0;1,0>:d   
        add3 (32|M0)             r22.0<1>:d    r14.0<1;0>:d      r2.6<0;0>:d       3:w              
        mov (16|M0)              r76.0<2>:ud   r30.0<1;1,0>:ud                  {Compacted}
        shl (16|M0)              r116.0<1>:q   r36.0<2;1,0>:d    1:w               {@4,$6.src}
        add (32|M0)              r20.0<1>:d    r14.0<1;1,0>:d    r2.7<0;1,0>:d    {Compacted,I@4}
(W)     send.ugm (1|M0)          null     r4  r116:2  0x83400000:a0.2        0x4200E504           {I@2,$7} // wr:1+2, rd:0; store.ugm.d32x32t.a32.ss[a0.2][A-0xF980]
        mov (16|M0)              r92.0<2>:ud   r22.0<1;1,0>:ud                  {Compacted}
        shl (16|M0)              r116.0<1>:q   r76.0<2;1,0>:d    1:w               {$7.src}
        add3 (32|M0)             r8.0<1>:d     r14.0<1;0>:d      r2.7<0;0>:d       1:w              
(W)     send.ugm (1|M0)          null     r4  r116:2  0x83800000:a0.2        0x4200E504           {I@2,$8} // wr:1+2, rd:0; store.ugm.d32x32t.a32.ss[a0.2][A-0xF900]
        mov (16|M0)              r96.0<2>:ud   r20.0<1;1,0>:ud                  {Compacted}
        shl (16|M0)              r116.0<1>:q   r92.0<2;1,0>:d    1:w               {$8.src}
(W)     send.ugm (1|M0)          null     r4  r116:2  0x83C00000:a0.2        0x4200E504           {I@1,$9} // wr:1+2, rd:0; store.ugm.d32x32t.a32.ss[a0.2][A-0xF880]
        mov (16|M0)              r100.0<2>:ud  r8.0<1;1,0>:ud                   {Compacted}
        shl (16|M0)              r116.0<1>:q   r96.0<2;1,0>:d    1:w               {$9.src}
(W)     send.ugm (1|M0)          null     r4  r116:2  0x84000000:a0.2        0x4200E504           {I@1,$10} // wr:1+2, rd:0; store.ugm.d32x32t.a32.ss[a0.2][A-0xF800]
        shl (16|M0)              r116.0<1>:q   r100.0<2;1,0>:d   1:w               {$10.src}
(W)     send.ugm (1|M0)          null     r4  r116:2  0x84400000:a0.2        0x4200E504           {I@1,$11} // wr:1+2, rd:0; store.ugm.d32x32t.a32.ss[a0.2][A-0xF780]
        sync.nop                             null                             {Compacted,$11.src}
(W)     send.ugm (1|M0)          r116     r4  null:0  0x83000000:a0.2        0x4220E500           {$12} // wr:1+0, rd:2; load.ugm.d32x32t.a32.ss[a0.2][A-0xFA00]
        add3 (32|M0)             r6.0<1>:d     r14.0<1;0>:d      r2.7<0;0>:d       2:w              
        add3 (32|M0)             r10.0<1>:d    r14.0<1;0>:d      r2.7<0;0>:d       3:w              
        mov (16|M0)              r104.0<2>:ud  r6.0<1;1,0>:ud                   {Compacted,I@2}
        mov (16|M0)              r5.0<2>:ud    r10.0<1;1,0>:ud                  {Compacted,I@2}
(W)     send.ugm (1|M0)          null     r4  r5:2  0x82400000:a0.2        0x4200E504           {I@1,$13} // wr:1+2, rd:0; store.ugm.d32x32t.a32.ss[a0.2][A-0xFB80]
        mov (16|M16)             r106.0<2>:ud  r9.0<1;1,0>:ud                   {Compacted}
        mov (16|M16)             r5.0<2>:ud    r11.0<1;1,0>:ud                  {Compacted,$13.src}
        shl (16|M0)              r9.0<1>:q     r34.0<2;1,0>:d    1:w              
        shl (16|M16)             r11.0<1>:q    r18.0<2;1,0>:d    1:w              
(W)     send.ugm (1|M0)          null     r4  r9:4  0x80000000:a0.2        0x4200F504           {I@1,$14} // wr:1+4, rd:0; store.ugm.d32x64t.a32.ss[a0.2][A-0x10000]
        shl (16|M0)              r9.0<1>:q     r46.0<2;1,0>:d    1:w               {$14.src}
        shl (16|M16)             r11.0<1>:q    r44.0<2;1,0>:d    1:w              
(W)     send.ugm (1|M0)          null     r4  r9:4  0x80800000:a0.2        0x4200F504           {I@1,$15} // wr:1+4, rd:0; store.ugm.d32x64t.a32.ss[a0.2][A-0xFF00]
        shl (16|M0)              r9.0<1>:q     r48.0<2;1,0>:d    1:w               {$15.src}
        shl (16|M0)              r11.0<1>:q    r16.0<2;1,0>:d    1:w              
        mov (16|M16)             r82.0<2>:ud   r31.0<1;1,0>:ud                  {Compacted}
(W)     send.ugm (1|M0)          null     r4  r9:4  0x81000000:a0.2        0x4200F504           {I@2,$0} // wr:1+4, rd:0; store.ugm.d32x64t.a32.ss[a0.2][A-0xFE00]
        shl (16|M0)              r9.0<1>:q     r40.0<2;1,0>:d    1:w               {$0.src}
        mov (16|M16)             r108.0<2>:ud  r7.0<1;1,0>:ud                   {Compacted}
        shl (16|M16)             r7.0<1>:q     r42.0<2;1,0>:d    1:w              
        shl (16|M16)             r42.0<1>:q    r82.0<2;1,0>:d    1:w               {I@4}
(W)     send.ugm (1|M0)          null     r4  r9:2  0x82000000:a0.2        0x4200E504           {I@4,$1} // wr:1+2, rd:0; store.ugm.d32x32t.a32.ss[a0.2][A-0xFC00]
        mov (16|M16)             r78.0<2>:ud   r33.0<1;1,0>:ud                  {Compacted}
(W)     send.ugm (1|M0)          null     r4  r5:4  0x81800000:a0.2        0x4200F504           {I@3,$2} // wr:1+4, rd:0; store.ugm.d32x64t.a32.ss[a0.2][A-0xFD00]
        shl (16|M16)             r46.0<1>:q    r78.0<2;1,0>:d    1:w               {I@1}
        shl (16|M16)             r5.0<1>:q     r60.0<2;1,0>:d    1:w               {$2.src}
        add (16|M16)             r92.0<1>:q    r116.0<1;1,0>:q   r1.5<0;1,0>:q    {Compacted,$12.dst}
(W)     send.ugm (1|M0)          r116     r4  null:0  0x83400000:a0.2        0x4220E500           {I@1,$3} // wr:1+0, rd:2; load.ugm.d32x32t.a32.ss[a0.2][A-0xF980]
        shl (16|M0)              r7.0<1>:q     r38.0<2;1,0>:d    1:w              
        shl (16|M16)             r74.0<1>:q    r64.0<2;1,0>:d    1:w              
        shl (16|M16)             r110.0<1>:q   r106.0<2;1,0>:d   1:w              
        shl (16|M16)             r64.0<1>:q    r108.0<2;1,0>:d   1:w              
(W)     send.ugm (1|M0)          null     r4  r5:4  0x82800000:a0.2        0x4200F504           {I@4,$4} // wr:1+4, rd:0; store.ugm.d32x64t.a32.ss[a0.2][A-0xFB00]
        sync.nop                             null                             {Compacted,$4.src}
(W)     send.ugm (1|M0)          r5       r4  null:0  0x82400000:a0.2        0x4220E500           {$5} // wr:1+0, rd:2; load.ugm.d32x32t.a32.ss[a0.2][A-0xFB80]
        add (32|M0)              r28.0<1>:d    r14.0<1;1,0>:d    r2.6<0;1,0>:d    {Compacted}
        mov (16|M16)             r86.0<2>:ud   r29.0<1;1,0>:ud                  {Compacted,I@1}
        shl (16|M16)             r38.0<1>:q    r86.0<2;1,0>:d    1:w               {I@1}
        sync.nop                             null                             {Compacted,$1.src}
(W)     send.ugm (1|M0)          r9       r4  null:0  0x80000000:a0.2        0x4240F500           {$6} // wr:1+0, rd:4; load.ugm.d32x64t.a32.ss[a0.2][A-0x10000]
        mov (16|M16)             r102.0<2>:ud  r21.0<1;1,0>:ud                  {Compacted}
        add (16|M16)             r76.0<1>:q    r38.0<1;1,0>:q    r1.5<0;1,0>:q    {Compacted,I@2}
        shl (16|M0)              r114.0<1>:q   r104.0<2;1,0>:d   1:w              
        shl (16|M16)             r16.0<1>:q    r102.0<2;1,0>:d   1:w               {I@3}
        add3 (32|M0)             r26.0<1>:d    r14.0<1;0>:d      r2.6<0;0>:d       1:w              
        add3 (32|M0)             r24.0<1>:d    r14.0<1;0>:d      r2.6<0;0>:d       2:w              
        mov (16|M0)              r84.0<2>:ud   r26.0<1;1,0>:ud                  {Compacted,I@2}
        mov (16|M16)             r94.0<2>:ud   r25.0<1;1,0>:ud                  {Compacted,I@2}
        shl (16|M0)              r44.0<1>:q    r84.0<2;1,0>:d    1:w               {I@2}
        shl (16|M16)             r34.0<1>:q    r94.0<2;1,0>:d    1:w               {I@2}
        mov (16|M16)             r98.0<2>:ud   r23.0<1;1,0>:ud                  {Compacted}
        add (16|M16)             r84.0<1>:q    r46.0<1;1,0>:q    r1.5<0;1,0>:q    {Compacted}
        add (16|M0)              r46.0<1>:q    r44.0<1;1,0>:q    r1.5<0;1,0>:q    {Compacted,I@4}
        add (16|M0)              r82.0<1>:q    r116.0<1;1,0>:q   r1.5<0;1,0>:q    {Compacted,$3.dst}
(W)     send.ugm (1|M0)          r116     r4  null:0  0x83800000:a0.2        0x4220E500           {I@1,$7} // wr:1+0, rd:2; load.ugm.d32x32t.a32.ss[a0.2][A-0xF900]
        add (16|M16)             r44.0<1>:q    r34.0<1;1,0>:q    r1.5<0;1,0>:q    {Compacted}
        shl (16|M16)             r18.0<1>:q    r98.0<2;1,0>:d    1:w              
        mov (16|M0)              r80.0<2>:ud   r28.0<1;1,0>:ud                  {Compacted}
        mov (16|M16)             r90.0<2>:ud   r27.0<1;1,0>:ud                  {Compacted}
        shl (16|M0)              r112.0<1>:q   r5.0<2;1,0>:d     1:w               {$5.dst}
(W)     send.ugm (1|M0)          r5       r4  null:0  0x81800000:a0.2        0x4240F500           {I@1,$8} // wr:1+0, rd:4; load.ugm.d32x64t.a32.ss[a0.2][A-0xFD00]
        mov (16|M0)              r88.0<2>:ud   r24.0<1;1,0>:ud                  {Compacted}
        shl (16|M0)              r48.0<1>:q    r80.0<2;1,0>:d    1:w              
        shl (16|M16)             r36.0<1>:q    r90.0<2;1,0>:d    1:w              
        add (16|M0)              r106.0<1>:q   r9.0<1;1,0>:q     r1.5<0;1,0>:q    {Compacted,$6.dst}
        add (16|M16)             r108.0<1>:q   r11.0<1;1,0>:q    r1.5<0;1,0>:q    {Compacted}
(W)     send.ugm (1|M0)          r9       r4  null:0  0x80800000:a0.2        0x4240F500           {I@1,$9} // wr:1+0, rd:4; load.ugm.d32x64t.a32.ss[a0.2][A-0xFF00]
        shl (16|M0)              r40.0<1>:q    r88.0<2;1,0>:d    1:w              
        add (16|M16)             r88.0<1>:q    r74.0<1;1,0>:q    r1.5<0;1,0>:q    {Compacted}
        add (16|M0)              r74.0<1>:q    r48.0<1;1,0>:q    r1.5<0;1,0>:q    {Compacted}
        add (16|M16)             r48.0<1>:q    r36.0<1;1,0>:q    r1.5<0;1,0>:q    {Compacted}
        add (16|M16)             r80.0<1>:q    r42.0<1;1,0>:q    r1.5<0;1,0>:q    {Compacted}
        add (16|M0)              r42.0<1>:q    r40.0<1;1,0>:q    r1.5<0;1,0>:q    {Compacted,I@5}
        send.ugm (32|M0)         r58      r46  null:0  0x0            0x08200B80           {A@3,$10} // wr:4+0, rd:2; load.ugm.d16u32.a64
        add (16|M16)             r40.0<1>:q    r18.0<1;1,0>:q    r1.5<0;1,0>:q    {Compacted}
        send.ugm (32|M0)         r46      r42  null:0  0x0            0x08200B80           {I@2,$11} // wr:4+0, rd:2; load.ugm.d16u32.a64
        add (16|M16)             r36.0<1>:q    r16.0<1;1,0>:q    r1.5<0;1,0>:q    {Compacted}
        add (16|M16)             r18.0<1>:q    r110.0<1;1,0>:q   r1.5<0;1,0>:q    {Compacted}
        send.ugm (32|M0)         r50      r82  null:0  0x0            0x08200B80           {$12} // wr:4+0, rd:2; load.ugm.d16u32.a64
        add (16|M0)              r78.0<1>:q    r116.0<1;1,0>:q   r1.5<0;1,0>:q    {Compacted,$7.dst}
(W)     send.ugm (1|M0)          r116     r4  null:0  0x83C00000:a0.2        0x4220E500           {I@1,$13} // wr:1+0, rd:2; load.ugm.d32x32t.a32.ss[a0.2][A-0xF880]
        send.ugm (32|M0)         r30      r106  null:0  0x0            0x08200B80           {$14} // wr:4+0, rd:2; load.ugm.d16u32.a64
        send.ugm (32|M0)         r54      r74  null:0  0x0            0x08200B80           {$15} // wr:4+0, rd:2; load.ugm.d16u32.a64
        shl (16|M16)             r60.0<1>:q    r5.0<2;1,0>:d     1:w               {$8.dst}
        add (16|M16)             r100.0<1>:q   r7.0<1;1,0>:q     r1.5<0;1,0>:q    {Compacted}
(W)     send.ugm (1|M0)          r5       r4  null:0  0x82800000:a0.2        0x4240F500           {I@1,$0} // wr:1+0, rd:4; load.ugm.d32x64t.a32.ss[a0.2][A-0xFB00]
        send.ugm (32|M0)         r52      r78  null:0  0x0            0x08200B80           {$1} // wr:4+0, rd:2; load.ugm.d16u32.a64
        add (16|M0)              r102.0<1>:q   r9.0<1;1,0>:q     r1.5<0;1,0>:q    {Compacted,$9.dst}
        add (16|M16)             r104.0<1>:q   r11.0<1;1,0>:q    r1.5<0;1,0>:q    {Compacted}
(W)     send.ugm (1|M0)          r9       r4  null:0  0x81000000:a0.2        0x4240F500           {I@1,$2} // wr:1+0, rd:4; load.ugm.d32x64t.a32.ss[a0.2][A-0xFE00]
        send.ugm (32|M0)         r28      r102  null:0  0x0            0x08200B80           {$3} // wr:4+0, rd:2; load.ugm.d16u32.a64
(W)     mov (1|M0)               r1.7<1>:uq    r1.5<0;1,0>:uq                  
        add (16|M0)              r38.0<1>:q    r116.0<1;1,0>:q   r1.5<0;1,0>:q    {Compacted,$13.dst}
(W)     send.ugm (1|M0)          r116     r4  null:0  0x84000000:a0.2        0x4220E500           {I@1,$4} // wr:1+0, rd:2; load.ugm.d32x32t.a32.ss[a0.2][A-0xF800]
        send.ugm (32|M0)         r42      r38  null:0  0x0            0x08200B80           {$5} // wr:4+0, rd:2; load.ugm.d16u32.a64
        shl (16|M0)              r32.0<1>:ud   r30.0<2;1,0>:uw   0x10:uw              {$14.dst}
        shl (16|M16)             r33.0<1>:ud   r31.0<2;1,0>:uw   0x10:uw             
        add (16|M16)             r96.0<1>:q    r5.0<1;1,0>:q     r1.5<0;1,0>:q    {Compacted,$0.dst}
        add (16|M0)              r86.0<1>:q    r7.0<1;1,0>:q     r1.5<0;1,0>:q    {Compacted}
        add (16|M0)              r5.0<1>:q     r112.0<1;1,0>:q   r1.5<0;1,0>:q    {Compacted}
        add (16|M16)             r7.0<1>:q     r60.0<1;1,0>:q    r1.5<0;1,0>:q    {Compacted}
        send.ugm (32|M0)         r20      r86  null:0  0x0            0x08200B80           {I@3,$6} // wr:4+0, rd:2; load.ugm.d16u32.a64
        add (16|M0)              r98.0<1>:q    r9.0<1;1,0>:q     r1.5<0;1,0>:q    {Compacted,$2.dst}
(W)     send.ugm (1|M0)          r9       r4  null:0  0x82000000:a0.2        0x4220E500           {I@1,$7} // wr:1+0, rd:2; load.ugm.d32x32t.a32.ss[a0.2][A-0xFC00]
        add (16|M0)              r94.0<1>:q    r11.0<1;1,0>:q    r1.5<0;1,0>:q    {Compacted}
        add (16|M16)             r11.0<1>:q    r64.0<1;1,0>:q    r1.5<0;1,0>:q    {Compacted}
        send.ugm (32|M0)         r26      r98  null:0  0x0            0x08200B80           {$8} // wr:4+0, rd:2; load.ugm.d16u32.a64
        send.ugm (32|M0)         r24      r94  null:0  0x0            0x08200B80           {I@2,$9} // wr:4+0, rd:2; load.ugm.d16u32.a64
        shl (16|M0)              r30.0<1>:ud   r28.0<2;1,0>:uw   0x10:uw              {$3.dst}
        shl (16|M16)             r31.0<1>:ud   r29.0<2;1,0>:uw   0x10:uw             
        add (16|M0)              r34.0<1>:q    r116.0<1;1,0>:q   r1.5<0;1,0>:q    {Compacted,$4.dst}
(W)     send.ugm (1|M0)          r116     r4  null:0  0x84400000:a0.2        0x4220E500           {I@1,$13} // wr:1+0, rd:2; load.ugm.d32x32t.a32.ss[a0.2][A-0xF780]
        send.ugm (32|M0)         r38      r34  null:0  0x0            0x08200B80           {$14} // wr:4+0, rd:2; load.ugm.d16u32.a64
        sync.nop                             null                             {Compacted,F@5}
        shl (16|M0)              r62.0<1>:ud   r42.0<2;1,0>:uw   0x10:uw              {$5.dst}
        shl (16|M16)             r63.0<1>:ud   r43.0<2;1,0>:uw   0x10:uw             
        sync.nop                             null                             {Compacted,$13.src}
(W)     mov (1|M0)               a0.2<1>:ud    r1.8<0;1,0>:ud                   {$7.src}
        add (16|M0)              r90.0<1>:q    r9.0<1;1,0>:q     r1.5<0;1,0>:q    {Compacted,$7.dst}
        add (16|M0)              r9.0<1>:q     r114.0<1;1,0>:q   r1.5<0;1,0>:q    {Compacted}
        send.ugm (32|M0)         r22      r90  null:0  0x0            0x08200B80           {I@2,$0} // wr:4+0, rd:2; load.ugm.d16u32.a64
        shl (16|M0)              r28.0<1>:ud   r26.0<2;1,0>:uw   0x10:uw              {$8.dst}
        shl (16|M16)             r29.0<1>:ud   r27.0<2;1,0>:uw   0x10:uw             
        shl (16|M0)              r26.0<1>:ud   r24.0<2;1,0>:uw   0x10:uw              {$9.dst}
        shl (16|M16)             r27.0<1>:ud   r25.0<2;1,0>:uw   0x10:uw             
        add (16|M0)              r16.0<1>:q    r116.0<1;1,0>:q   r1.5<0;1,0>:q    {Compacted,$13.dst}
        send.ugm (32|M0)         r34      r16  null:0  0x0            0x08200B80           {I@1,$2} // wr:4+0, rd:2; load.ugm.d16u32.a64
        send.ugm (32|M0)         r16      r9  null:0  0x0            0x08200B80           {$3} // wr:4+0, rd:2; load.ugm.d16u32.a64
        send.ugm (32|M0)         r10      r5  null:0  0x0            0x08200B80           {$4} // wr:4+0, rd:2; load.ugm.d16u32.a64
        sync.nop                             null                             {Compacted,F@4}
        shl (16|M0)              r66.0<1>:ud   r38.0<2;1,0>:uw   0x10:uw              {$14.dst}
        shl (16|M16)             r67.0<1>:ud   r39.0<2;1,0>:uw   0x10:uw             
        shl (16|M0)              r24.0<1>:ud   r22.0<2;1,0>:uw   0x10:uw              {$0.dst}
        shl (16|M16)             r25.0<1>:ud   r23.0<2;1,0>:uw   0x10:uw             
        shl (16|M0)              r22.0<1>:ud   r20.0<2;1,0>:uw   0x10:uw              {$6.dst}
        shl (16|M16)             r23.0<1>:ud   r21.0<2;1,0>:uw   0x10:uw             
        shl (16|M0)              r20.0<1>:ud   r50.0<2;1,0>:uw   0x10:uw              {$12.dst}
        shl (16|M16)             r21.0<1>:ud   r51.0<2;1,0>:uw   0x10:uw             
        shl (16|M0)              r50.0<1>:ud   r52.0<2;1,0>:uw   0x10:uw              {$1.dst}
        shl (16|M16)             r51.0<1>:ud   r53.0<2;1,0>:uw   0x10:uw             
        shl (16|M0)              r52.0<1>:ud   r54.0<2;1,0>:uw   0x10:uw              {$15.dst}
        shl (16|M16)             r53.0<1>:ud   r55.0<2;1,0>:uw   0x10:uw             
        shl (16|M0)              r54.0<1>:ud   r58.0<2;1,0>:uw   0x10:uw              {$10.dst}
        shl (16|M16)             r55.0<1>:ud   r59.0<2;1,0>:uw   0x10:uw             
        shl (16|M0)              r58.0<1>:ud   r46.0<2;1,0>:uw   0x10:uw              {$11.dst}
        shl (16|M16)             r59.0<1>:ud   r47.0<2;1,0>:uw   0x10:uw             
        sync.nop                             null                             {Compacted,F@3}
        shl (16|M0)              r68.0<1>:ud   r34.0<2;1,0>:uw   0x10:uw              {$2.dst}
        shl (16|M16)             r69.0<1>:ud   r35.0<2;1,0>:uw   0x10:uw             
        sync.nop                             null                             {Compacted,F@2}
        shl (16|M0)              r70.0<1>:ud   r16.0<2;1,0>:uw   0x10:uw              {$3.dst}
        shl (16|M16)             r71.0<1>:ud   r17.0<2;1,0>:uw   0x10:uw             
        sync.nop                             null                             {Compacted,F@1}
        shl (16|M0)              r72.0<1>:ud   r10.0<2;1,0>:uw   0x10:uw              {$4.dst}
        shl (16|M16)             r73.0<1>:ud   r11.0<2;1,0>:uw   0x10:uw             
L4928:
(W)     shl (1|M0)               r2.2<1>:q     r3.5<0;1,0>:q     2:w              
        or (32|M0)               r82.0<1>:d    r14.0<1;1,0>:d    1:w               {Compacted}
(W)     add (1|M0)               r6.0<1>:q     r2.2<0;1,0>:q     r2.5<0;1,0>:q    {Compacted,I@2}
        or (32|M0)               r84.0<1>:d    r14.0<1;1,0>:d    2:w               {Compacted}
(W)     send.ugm (1|M0)          r5       r6  null:0  0x0            0x02109580           {I@2,$5} // wr:1+0, rd:1; load.ugm.d32x2t.a64
        or (32|M0)               r86.0<1>:d    r14.0<1;1,0>:d    3:w               {Compacted}
(W)     mov (2|M0)               r1.12<1>:d    r5.0<1;1,0>:d                    {$5.dst}
(W)     cmp (32|M0)   (lt)f1.0   null<1>:d     r1.12<0;1,0>:d    r1.13<0;1,0>:d   {I@1}
(W&~f1.0) jmpi                               L10976                                
L5040:
(W)     cmp (32|M0)   (eq)f0.0   null<1>:d     r3.9<0;1,0>:d     0:w              
(W&~f0.0) jmpi                               L5104                                
L5072:
(W)     mov (1|M0)               r2.11<1>:d    -1:w                              
(W)     jmpi                                 L5592                                
L5104:
(W)     asr (1|M0)               r2.7<1>:d     r3.9<0;1,0>:d     31:w              
(W)     add (1|M0)               r2.4<1>:d     r2.7<0;1,0>:d     r3.9<0;1,0>:d    {I@1}
(W)     xor (1|M0)               r2.6<1>:d     r2.4<0;1,0>:d     r2.7<0;1,0>:d    {I@1}
(W)     xor (1|M0)               cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x30:uw              {A@1}
(W)     mov (1|M0)               r2.12<1>:f    r2.6<0;1,0>:ud                   {A@1}
(W)     mov (1|M0)               r2.10<1>:f    r3.6<0;1,0>:ud                  
(W)     mov (1|M0)               r2.4<1>:ud    r2.12<0;1,0>:f                   {F@2}
(W)     math.inv (1|M0)          r2.5<1>:f     r2.12<0;1,0>:f                  
(W)     add (1|M0)               r2.8<1>:d     r2.6<0;1,0>:d     -r2.4<0;1,0>:d   {I@1}
(W)     mov (1|M0)               r2.4<1>:f     0xB4C00000:f                               {Compacted,I@1}
(W)     mad (1|M0)               r2.14<1>:f    r2.5<0;0>:f       r2.4<0;0>:f       r2.5<0>:f        {A@1}
(W)     mov (1|M0)               r2.4<1>:ud    r2.10<0;1,0>:f                   {F@1}
(W)     mul (1|M0)               r2.5<1>:f     r2.10<0;1,0>:f    r2.14<0;1,0>:f  
(W)     add (1|M0)               r2.9<1>:d     r3.6<0;1,0>:d     -r2.4<0;1,0>:d   {I@1}
(W)     mov (1|M0)               r2.13<1>:ud   r2.5<0;1,0>:f                    {F@1}
(W)     mov (1|M0)               r2.4<1>:f     r2.8<0;1,0>:ud                   {I@2}
(W)     mov (1|M0)               r2.5<1>:f     r2.9<0;1,0>:ud                   {I@1}
(W)     mov (1|M0)               r2.8<1>:f     r2.13<0;1,0>:ud                 
(W)     mad (1|M0)               r2.9<1>:f     r2.10<0;0>:f      r2.8<0;0>:f       -r2.12<0>:f      {F@1}
(W)     mad (1|M0)               r2.4<1>:f     r2.5<0;0>:f       r2.8<0;0>:f       -r2.4<0>:f      
(W)     add (1|M0)               r2.4<1>:f     r2.9<0;1,0>:f     r2.4<0;1,0>:f    {F@1}
(W)     mul (1|M0)               r2.4<1>:f     r2.14<0;1,0>:f    r2.4<0;1,0>:f    {F@1}
(W)     xor (1|M0)               cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x30:uw              {A@1}
(W)     mov (1|M0)               r2.4<1>:ud    r2.4<0;1,0>:f                    {A@1}
(W)     add (1|M0)               r2.5<1>:d     r2.4<0;1,0>:d     r2.13<0;1,0>:d   {I@1}
(W)     mul (1|M0)               acc0.0<1>:d   r2.5<0;1,0>:d     r2.12<0;1,0>:uw  {I@1}
(W)     macl (1|M0)              r3.0<1>:d     r2.5<0;1,0>:d     r2.6<0;1,0>:d   
(W)     add (1|M0)               r2.4<1>:d     r3.6<0;1,0>:d     -r3.0<0;1,0>:d   {I@1}
(W)     cmp (1|M0)    (ge)f1.0   r2.4<1>:ud    r2.4<0;1,0>:ud    r2.6<0;1,0>:ud   {I@1}
(W)     add3 (1|M0)              r2.4<1>:d     r2.5<0;0>:d       r2.7<0;0>:d       -r2.4<0>:d       {I@1}
(W)     xor (1|M0)               r2.11<1>:d    r2.4<0;1,0>:d     r2.7<0;1,0>:d    {I@1}
L5592:
(W)     mov (1|M0)               r2.4<1>:f     r3.2<0;1,0>:d                    {I@1}
        cmp (32|M0)   (eq)f3.0   null<2>:w     r56.0<1;1,0>:w    0:w              
(W)     math.sqt (1|M0)          r3.15<1>:f    r2.4<0;1,0>:f                    {F@1}
(W)     mov (1|M0)               r3.14<1>:ud   0x3F317200:ud                             
(W)     mov (1|M0)               r3.13<1>:ud   0x35BFBE8E:ud                             
(W)     mov (1|M0)               r3.12<1>:ud   0xBF000000:ud                             
(W)     mov (1|M0)               r3.11<1>:ud   0x3EAAAA83:ud                             
(W)     mov (1|M0)               r3.10<1>:ud   0xBE7FFF78:ud                             
(W)     mov (1|M0)               r3.9<1>:ud    0x3E4CE814:ud                             
(W)     mov (1|M0)               r3.0<1>:f     0xBE2ACEE6:f                              
(W)     mov (1|M0)               r2.15<1>:f    1.400587e-01:f                              
(W)     mov (1|M0)               r2.14<1>:f    0xBDF9889E:f                              
(W)     mov (1|M0)               r2.13<1>:f    0xBE0402C8:f                              
(W)     mov (1|M0)               r2.12<1>:f    0x3E0F335D:f                              
(W)     mov (1|M0)               r2.10<1>:f    1.0:f                              
L5832:
(W)     mul (1|M0)               acc0.0<1>:d   r1.12<0;1,0>:d    r3.6<0;1,0>:uw  
(W)     macl (1|M0)              r5.0<1>:d     r1.12<0;1,0>:d    r3.3<0;1,0>:d    {Compacted}
(W)     add (1|M0)               r2.7<1>:d     r5.0<0;1,0>:d     r3.6<0;1,0>:d    {I@1}
(W)     shl (1|M0)               r4.2<1>:q     r2.7<0;1,0>:d     1:w               {I@1}
(W)     add (1|M0)               r8.0<1>:q     r4.2<0;1,0>:q     r2.0<0;1,0>:q    {Compacted,@1,$6.src}
(W)     add (1|M0)               r6.0<1>:q     r4.2<0;1,0>:q     r2.1<0;1,0>:q    {Compacted}
(W)     send.ugm (1|M0)          r7       r8  null:0  0x0            0x04100B80           {I@2,$7} // wr:2+0, rd:1; load.ugm.d16u32.a64
(W)     send.ugm (1|M0)          r5       r6  null:0  0x0            0x04100B80           {I@1,$7} // wr:2+0, rd:1; load.ugm.d16u32.a64
(W)     shl (1|M0)               r4.1<1>:ud    r7.0<0;1,0>:uw    0x10:uw             
(W)     shl (1|M0)               r4.4<1>:ud    r5.0<0;1,0>:uw    0x10:uw              {$7.dst}
(W)     mul (1|M0)               r4.5<1>:f     r4.1<0;1,0>:f     -1.442695e+00:f               {I@2}
(W)     add (1|M0)               r4.4<1>:f     r4.4<0;1,0>:f     r4.3<0;1,0>:f    {I@1}
(W)     rndz (1|M0)              r5.0<1>:f     r4.5<0;1,0>:f                    {F@2}
(W)     mov (1|M0)               r4.10<1>:bf   r4.4<0;1,0>:f                    {F@2}
(W)     mad (1|M0)               r4.4<1>:f     -r4.1<0;0>:f      r3.8<0;0>:f       r5.0<0>:f        {F@2}
(W)     math.exp (1|M0)          r6.0<1>:f     r5.0<0;1,0>:f                   
(W)     mad (1|M0)               r4.4<1>:f     r4.4<0;0>:f       r3.7<0;0>:f       r5.0<0>:f        {F@1}
(W)     cmp (1|M0)    (gt)f2.0   null<1>:f     r4.1<0;1,0>:f     105.0:f              
(W)     mul (1|M0)               r5.1<1>:f     r4.4<0;1,0>:f     1.442695e+00:f               {F@2}
(W)     cmp (1|M0)    (lt)f1.0   null<1>:f     r4.1<0;1,0>:f     -105.0:f              
(W)     math.exp (1|M0)          r6.1<1>:f     r5.1<0;1,0>:f                    {F@2}
(W)     mad (1|M0)               r4.4<1>:f     r2.10<0;0>:f      r6.0<0;0>:f       r6.1<0>:f        {M@1}
(W)     shl (1|M0)               r4.1<1>:ud    r4.10<0;1,0>:uw   0x10:uw              {F@2}
(W)     math.inv (1|M0)          r4.4<1>:f     r4.4<0;1,0>:f                    {F@1}
(W)     cmp (32|M0)   (lt)f0.0   null<1>:f     r4.1<0;1,0>:f     20.0:f               {I@1}
(W&~f2.0) sel (1|M0)             r4.4<1>:f     r4.4<0;1,0>:f     1.0:f               {M@1}
(W&~f1.0) sel (1|M0)             r2.6<1>:f     r4.4<0;1,0>:f     0.0:f               {F@1}
(W&~f0.0) jmpi                               L6920                                
L6256:
(W)     mul (1|M0)               r4.4<1>:f     r4.1<0;1,0>:f     1.442695e+00:f              
(W)     cmp (1|M0)    (lt)f2.0   null<1>:f     r4.1<0;1,0>:f     -105.0:f              
(W)     rndz (1|M0)              r5.0<1>:f     r4.4<0;1,0>:f                    {Compacted,F@2}
(W)     cmp (1|M0)    (gt)f1.0   null<1>:f     r4.1<0;1,0>:f     105.0:f              
(W)     mad (1|M0)               r4.4<1>:f     r4.1<0;0>:f       r3.8<0;0>:f       r5.0<0>:f        {F@2}
(W)     math.exp (1|M0)          r6.0<1>:f     r5.0<0;1,0>:f                   
(W)     mad (1|M0)               r4.4<1>:f     r4.4<0;0>:f       r3.7<0;0>:f       r5.0<0>:f        {F@1}
(W)     mul (1|M0)               r5.1<1>:f     r4.4<0;1,0>:f     1.442695e+00:f               {F@1}
(W)     math.exp (1|M0)          r6.1<1>:f     r5.1<0;1,0>:f                    {F@1}
(W)     mad (1|M0)               r4.4<1>:f     r2.10<0;0>:f      r6.0<0;0>:f       r6.1<0>:f        {M@1}
(W&~f2.0) sel (1|M0)             r4.4<1>:f     r4.4<0;1,0>:f     1.0:f               {F@1}
(W&~f1.0) sel (1|M0)             r4.4<1>:f     r4.4<0;1,0>:f     inf:f               {F@1}
(W)     cmp (32|M0)   (gt)f1.0   null<1>:f     r4.4<0;1,0>:f     0.0:f               {F@1}
(W)     and (1|M0)               r4.5<1>:d     r4.4<0;1,0>:d     2147483647:d              
(W&f1.0) cmp (32|M0)  (lt)f1.0   null<1>:f     r4.5<0;1,0>:f     inf:f               {I@1}
(W&f1.0) jmpi                                L6536                                
L6504:
(W)     math.log (1|M0)          r4.1<1>:f     r4.4<0;1,0>:f                   
(W)     jmpi                                 L6920                                
L6536:
(W)     cmp (32|M0)   (lt)f2.0   null<1>:f     r4.4<0;1,0>:f     0x800000:f              
(W)     mul (1|M0)               r4.6<1>:f     r4.4<0;1,0>:f     8.388608e+06:f              
(W)     mov (1|M0)               r4.5<1>:ud    0xC1B80000:ud                              {F@3}
(W&f2.0) sel (1|M0)              r4.4<1>:f     r4.6<0;1,0>:f     r4.4<0;1,0>:f    {A@1}
(W&f2.0) sel (1|M0)              r4.7<1>:f     r4.5<0;1,0>:f     0.0:f               {I@1}
(W)     add (1|M0)               r4.4<1>:d     r4.4<0;1,0>:d     -1059760811:d               {F@2}
(W)     and (1|M0)               r4.5<1>:d     r4.4<0;1,0>:d     8388607:d               {A@1}
(W)     asr (1|M0)               r4.4<1>:d     r4.4<0;1,0>:d     23:w              
(W)     add (1|M0)               r4.6<1>:d     r4.5<0;1,0>:d     1059760811:d               {I@2}
(W)     mov (1|M0)               r4.4<1>:f     r4.4<0;1,0>:d                    {I@2}
(W)     add (1|M0)               r4.5<1>:f     r4.7<0;1,0>:f     r4.4<0;1,0>:f    {A@1}
(W)     add (1|M0)               r4.4<1>:f     r4.6<0;1,0>:f     -1.0:f              
(W)     mad (1|M0)               r4.6<1>:f     r2.12<0;0>:f      r2.13<0;0>:f      r4.4<0>:f        {F@1}
(W)     mad (1|M0)               r4.6<1>:f     r2.14<0;0>:f      r4.4<0;0>:f       r4.6<0>:f        {F@1}
(W)     mad (1|M0)               r4.6<1>:f     r2.15<0;0>:f      r4.4<0;0>:f       r4.6<0>:f        {F@1}
(W)     mad (1|M0)               r4.6<1>:f     r3.0<0;0>:f       r4.4<0;0>:f       r4.6<0>:f        {F@1}
(W)     mad (1|M0)               r4.6<1>:f     r3.9<0;0>:f       r4.4<0;0>:f       r4.6<0>:f        {F@1}
(W)     mad (1|M0)               r4.6<1>:f     r3.10<0;0>:f      r4.4<0;0>:f       r4.6<0>:f        {F@1}
(W)     mad (1|M0)               r4.6<1>:f     r3.11<0;0>:f      r4.4<0;0>:f       r4.6<0>:f        {F@1}
(W)     mad (1|M0)               r4.6<1>:f     r3.12<0;0>:f      r4.4<0;0>:f       r4.6<0>:f        {F@1}
(W)     mul (1|M0)               r4.6<1>:f     r4.4<0;1,0>:f     r4.6<0;1,0>:f    {F@1}
(W)     mad (1|M0)               r4.4<1>:f     r4.4<0;0>:f       r4.4<0;0>:f       r4.6<0>:f        {F@1}
(W)     mad (1|M0)               r4.4<1>:f     r4.4<0;0>:f       r3.13<0;0>:f      r4.5<0>:f        {F@1}
(W)     mad (1|M0)               r4.1<1>:f     r4.4<0;0>:f       r3.14<0;0>:f      r4.5<0>:f        {F@1}
L6920:
(W)     mul (1|M0)               acc0.0<1>:d   r1.12<0;1,0>:d    r3.2<0;1,0>:uw  
(W)     macl (1|M0)              r5.0<1>:d     r1.12<0;1,0>:d    r3.1<0;1,0>:d    {Compacted}
(W)     mul (1|M0)               r2.4<1>:f     r4.1<0;1,0>:f     -r4.2<0;1,0>:f   {F@1}
(W)     add (1|M0)               r4.1<1>:d     r5.0<0;1,0>:d     r2.11<0;1,0>:d   {Compacted,A@1}
(W)     mul (1|M0)               r4.4<1>:f     r2.4<0;1,0>:f     1.442695e+00:f              
(W)     mul (1|M0)               acc0.0<1>:d   r4.1<0;1,0>:d     r3.4<0;1,0>:uw   {I@1}
(W)     macl (1|M0)              r5.0<1>:d     r4.1<0;1,0>:d     r3.2<0;1,0>:d    {Compacted}
(W)     rndz (1|M0)              r2.5<1>:f     r4.4<0;1,0>:f                    {F@1}
        add (32|M0)              r10.0<1>:d    r5.0<0;1,0>:d     r82.0<1;1,0>:d   {Compacted,I@1}
        add (32|M0)              r8.0<1>:d     r5.0<0;1,0>:d     r14.0<1;1,0>:d   {Compacted}
        add (32|M0)              r6.0<1>:d     r5.0<0;1,0>:d     r84.0<1;1,0>:d   {Compacted}
        mov (16|M0)              r40.0<2>:ud   r10.0<1;1,0>:ud                  {Compacted,I@3}
        mov (16|M16)             r36.0<2>:ud   r11.0<1;1,0>:ud                  {Compacted}
        mov (16|M0)              r42.0<2>:ud   r8.0<1;1,0>:ud                   {Compacted,I@4}
        mov (16|M16)             r34.0<2>:ud   r9.0<1;1,0>:ud                   {Compacted}
        add (32|M0)              r16.0<1>:d    r5.0<0;1,0>:d     r86.0<1;1,0>:d   {Compacted}
        shl (16|M0)              r38.0<1>:q    r40.0<2;1,0>:d    1:w               {I@5}
        shl (16|M16)             r11.0<1>:q    r36.0<2;1,0>:d    1:w               {I@5}
        mov (16|M0)              r18.0<2>:ud   r6.0<1;1,0>:ud                   {Compacted}
        shl (16|M16)             r64.0<1>:q    r34.0<2;1,0>:d    1:w               {I@5}
        mov (16|M16)             r9.0<2>:ud    r7.0<1;1,0>:ud                   {Compacted}
        shl (16|M0)              r40.0<1>:q    r42.0<2;1,0>:d    1:w              
        add (16|M0)              r5.0<1>:q     r38.0<1;1,0>:q    r1.2<0;1,0>:q    {Compacted,I@6}
        add (16|M16)             r7.0<1>:q     r11.0<1;1,0>:q    r1.2<0;1,0>:q    {Compacted,I@6}
        mov (16|M0)              r36.0<2>:ud   r16.0<1;1,0>:ud                  {Compacted}
        shl (16|M16)             r60.0<1>:q    r9.0<2;1,0>:d     1:w               {I@5}
        shl (16|M0)              r42.0<1>:q    r18.0<2;1,0>:d    1:w              
        mov (16|M16)             r9.0<2>:ud    r17.0<1;1,0>:ud                  {Compacted}
        add (16|M16)             r18.0<1>:q    r64.0<1;1,0>:q    r1.2<0;1,0>:q    {Compacted}
        add (16|M0)              r16.0<1>:q    r40.0<1;1,0>:q    r1.2<0;1,0>:q    {Compacted,I@7}
        send.ugm (32|M0)         r34      r5  null:0  0x0            0x08200B80           {I@7,$8} // wr:4+0, rd:2; load.ugm.d16u32.a64
        shl (16|M0)              r48.0<1>:q    r36.0<2;1,0>:d    1:w               {I@6}
        send.ugm (32|M0)         r36      r16  null:0  0x0            0x08200B80           {I@1,$9} // wr:4+0, rd:2; load.ugm.d16u32.a64
        shl (16|M16)             r56.0<1>:q    r9.0<2;1,0>:d     1:w              
        add (16|M0)              r7.0<1>:q     r42.0<1;1,0>:q    r1.2<0;1,0>:q    {Compacted,$8.src}
        add (16|M16)             r9.0<1>:q     r60.0<1;1,0>:q    r1.2<0;1,0>:q    {Compacted}
        add (16|M0)              r5.0<1>:q     r38.0<1;1,0>:q    r1.1<0;1,0>:q    {Compacted}
        send.ugm (32|M0)         r38      r7  null:0  0x0            0x08200B80           {I@1,$10} // wr:4+0, rd:2; load.ugm.d16u32.a64
        add (16|M0)              r16.0<1>:q    r48.0<1;1,0>:q    r1.2<0;1,0>:q    {Compacted,$9.src}
        add (16|M16)             r18.0<1>:q    r56.0<1;1,0>:q    r1.2<0;1,0>:q    {Compacted}
        add (16|M16)             r7.0<1>:q     r11.0<1;1,0>:q    r1.1<0;1,0>:q    {Compacted,$10.src}
        add (16|M0)              r9.0<1>:q     r40.0<1;1,0>:q    r1.1<0;1,0>:q    {Compacted}
        send.ugm (32|M0)         r40      r16  null:0  0x0            0x08200B80           {I@1,$11} // wr:4+0, rd:2; load.ugm.d16u32.a64
        add (16|M16)             r11.0<1>:q    r64.0<1;1,0>:q    r1.1<0;1,0>:q    {Compacted}
        add (16|M0)              r16.0<1>:q    r42.0<1;1,0>:q    r1.1<0;1,0>:q    {Compacted,$11.src}
        send.ugm (32|M0)         r42      r5  null:0  0x0            0x08200B80           {I@1,$12} // wr:4+0, rd:2; load.ugm.d16u32.a64
        add (16|M0)              r5.0<1>:q     r48.0<1;1,0>:q    r1.1<0;1,0>:q    {Compacted,$12.src}
        send.ugm (32|M0)         r48      r9  null:0  0x0            0x08200B80           {I@1,$13} // wr:4+0, rd:2; load.ugm.d16u32.a64
        add (16|M16)             r18.0<1>:q    r60.0<1;1,0>:q    r1.1<0;1,0>:q    {Compacted}
        send.ugm (32|M0)         r10      r16  null:0  0x0            0x08200B80           {I@1,$14} // wr:4+0, rd:2; load.ugm.d16u32.a64
        add (16|M16)             r7.0<1>:q     r56.0<1;1,0>:q    r1.1<0;1,0>:q    {Compacted}
        send.ugm (32|M0)         r16      r5  null:0  0x0            0x08200B80           {I@1,$15} // wr:4+0, rd:2; load.ugm.d16u32.a64
(W)     mad (1|M0)               r4.1<1>:f     r2.4<0;0>:f       r3.8<0;0>:f       r2.5<0>:f        {F@1}
(W)     math.exp (1|M0)          r2.8<1>:f     r2.5<0;1,0>:f                   
(W)     mad (1|M0)               r4.1<1>:f     r4.1<0;0>:f       r3.7<0;0>:f       r2.5<0>:f        {F@1}
(W)     cmp (1|M0)    (lt)f1.0   null<1>:f     r2.4<0;1,0>:f     -105.0:f              
(W)     mul (1|M0)               r4.1<1>:f     r4.1<0;1,0>:f     1.442695e+00:f               {F@2}
(W)     cmp (1|M0)    (gt)f0.0   null<1>:f     r2.4<0;1,0>:f     105.0:f              
(W)     math.exp (1|M0)          r4.1<1>:f     r4.1<0;1,0>:f                    {F@2}
(W)     mul (1|M0)               r4.1<1>:f     r2.8<0;1,0>:f     r4.1<0;1,0>:f    {M@1}
        sync.nop                             null                             {Compacted,$14.src}
        shl (16|M16)             r18.0<1>:ud   r35.0<2;1,0>:uw   0x10:uw              {$8.dst}
(W&~f1.0) sel (1|M0)             r4.1<1>:f     r4.1<0;1,0>:f     0.0:f               {F@1}
        shl (16|M0)              r6.0<1>:ud    r34.0<2;1,0>:uw   0x10:uw              {$15.src}
(W&~f0.0) sel (1|M0)             r1.10<1>:f    r4.1<0;1,0>:f     inf:f               {F@1}
        mov (16|M0)              r7.0<1>:uw    r34.0<2;1,0>:uw                 
        mov (16|M16)             r5.0<1>:uw    r35.0<2;1,0>:uw                 
        sync.nop                             null                             {Compacted,$13.src}
        shl (16|M0)              r12.0<1>:ud   r36.0<2;1,0>:uw   0x10:uw              {$9.dst}
        mov (16|M0)              r75.0<1>:uw   r36.0<2;1,0>:uw                 
        mul (32|M0)              acc2.0<1>:f   r22.0<1;1,0>:f    r1.10<0;1,0>:f   {Compacted,F@1}
        mul (16|M0)              acc0.0<1>:f   r7.0<1;1,0>:bf    r6.0<1;1,0>:f    {I@4}
        mul (16|M16)             acc1.0<1>:f   r5.0<1;1,0>:bf    r18.0<1;1,0>:f   {I@3}
        mov (16|M16)             r74.0<1>:uw   r37.0<2;1,0>:uw                 
        shl (16|M16)             r18.0<1>:ud   r37.0<2;1,0>:uw   0x10:uw              {F@1}
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     r75.0<1;0>:bf     r12.0<1>:f       {I@3}
        mov (16|M0)              r35.0<1>:uw   r38.0<2;1,0>:uw                  {$10.dst}
        shl (16|M0)              r6.0<1>:ud    r38.0<2;1,0>:uw   0x10:uw             
        mad (16|M16)             acc1.0<1>:f   acc1.0<1;0>:f     r74.0<1;0>:bf     r18.0<1>:f       {I@3}
        shl (16|M16)             r19.0<1>:ud   r39.0<2;1,0>:uw   0x10:uw             
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     r35.0<1;0>:bf     r6.0<1>:f        {I@2}
        mov (16|M16)             r39.0<1>:uw   r39.0<2;1,0>:uw                 
        mov (16|M0)              r76.0<1>:uw   r40.0<2;1,0>:uw                  {$11.dst}
        shl (16|M0)              r12.0<1>:ud   r40.0<2;1,0>:uw   0x10:uw              {F@3}
        shl (16|M16)             r18.0<1>:ud   r41.0<2;1,0>:uw   0x10:uw              {F@2}
        mad (16|M16)             acc1.0<1>:f   acc1.0<1;0>:f     r39.0<1;0>:bf     r19.0<1>:f       {I@4}
        mov (16|M16)             r40.0<1>:uw   r41.0<2;1,0>:uw                 
(W)     mov (32|M0)              r44.0<1>:ud   0x0:ud                             
        shl (16|M16)             r36.0<1>:ud   r43.0<2;1,0>:uw   0x10:uw              {$12.dst}
        mov (16|M0)              r34.0<1>:uw   r42.0<2;1,0>:uw                 
        mov (16|M16)             r38.0<1>:uw   r43.0<2;1,0>:uw                 
        shl (16|M0)              r6.0<1>:ud    r42.0<2;1,0>:uw   0x10:uw              {F@2}
        mad (16|M16)             r45.0<1>:f    acc1.0<1;0>:f     r40.0<1;0>:bf     r18.0<1>:f       {I@5}
        mov (16|M0)              r43.0<1>:uw   r48.0<2;1,0>:uw                  {$13.dst}
        shl (16|M0)              r18.0<1>:ud   r48.0<2;1,0>:uw   0x10:uw              {F@1}
        mad (16|M0)              r44.0<1>:f    acc0.0<1;0>:f     r76.0<1;0>:bf     r12.0<1>:f      
        mul (16|M0)              acc0.0<1>:f   r34.0<1;1,0>:bf   r6.0<1;1,0>:f    {I@3}
        mul (16|M16)             acc1.0<1>:f   r38.0<1;1,0>:bf   r36.0<1;1,0>:f  
        shl (16|M16)             r19.0<1>:ud   r49.0<2;1,0>:uw   0x10:uw             
        mov (16|M16)             r42.0<1>:uw   r49.0<2;1,0>:uw                 
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     r43.0<1;0>:bf     r18.0<1>:f       {I@3}
        mov (16|M0)              r41.0<1>:uw   r10.0<2;1,0>:uw                  {$14.dst}
(W)     add (16|M0)              r37.0<1>:f    r44.0<1;1,0>:f    r45.0<1;1,0>:f   {Compacted,F@4}
        shl (16|M0)              r12.0<1>:ud   r10.0<2;1,0>:uw   0x10:uw             
        mad (16|M16)             acc1.0<1>:f   acc1.0<1;0>:f     r42.0<1;0>:bf     r19.0<1>:f       {I@3}
        shl (16|M16)             r36.0<1>:ud   r11.0<2;1,0>:uw   0x10:uw              {F@4}
(W)     mov (8|M0)               r45.0<1>:ud   r37.8<1;1,0>:ud                  {Compacted,F@2}
        mov (16|M16)             r44.0<1>:uw   r11.0<2;1,0>:uw                 
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     r41.0<1;0>:bf     r12.0<1>:f       {I@4}
        shl (16|M0)              r6.0<1>:ud    r16.0<2;1,0>:uw   0x10:uw              {$15.dst}
        shl (16|M16)             r10.0<1>:ud   r17.0<2;1,0>:uw   0x10:uw             
(W)     add (8|M0)               r11.0<1>:f    r37.0<1;1,0>:f    r45.0<1;1,0>:f   {Compacted,I@3}
        mad (16|M16)             acc1.0<1>:f   acc1.0<1;0>:f     r44.0<1;0>:bf     r36.0<1>:f      
        mov (16|M16)             r12.0<1>:uw   r17.0<2;1,0>:uw                  {F@3}
        mov (16|M0)              r45.0<1>:uw   r16.0<2;1,0>:uw                  {F@2}
(W)     mov (32|M0)              r46.0<1>:ud   0x0:ud                             
(W)     mov (4|M0)               r16.0<1>:ud   r11.4<1;1,0>:ud                  {Compacted}
        mad (16|M0)              r46.0<1>:f    acc0.0<1;0>:f     r45.0<1;0>:bf     r6.0<1>:f        {I@2}
        mad (16|M16)             r47.0<1>:f    acc1.0<1;0>:f     r12.0<1;0>:bf     r10.0<1>:f      
(W)     add (4|M0)               r10.0<1>:f    r11.0<1;1,0>:f    r16.0<1;1,0>:f   {Compacted,I@1}
(W)     add (16|M0)              r6.0<1>:f     r46.0<1;1,0>:f    r47.0<1;1,0>:f   {Compacted,F@2}
(W)     add (1|M0)               r4.4<1>:f     r10.0<0;1,0>:f    r10.2<0;1,0>:f   {Compacted,F@2}
(W)     add (1|M0)               r4.5<1>:f     r10.1<0;1,0>:f    r10.3<0;1,0>:f  
(W)     mov (8|M0)               r10.0<1>:ud   r6.8<1;1,0>:ud                   {Compacted,F@1}
(W)     add (1|M0)               r4.5<1>:f     r4.4<0;1,0>:f     r4.5<0;1,0>:f   
(W)     add (8|M0)               r6.0<1>:f     r6.0<1;1,0>:f     r10.0<1;1,0>:f   {Compacted,I@1}
(W)     add (1|M0)               r4.9<1>:f     r4.5<0;1,0>:f     1e-06:f               {F@2}
(W)     mov (4|M0)               r10.0<1>:ud   r6.4<1;1,0>:ud                   {Compacted,F@2}
(W)     math.rsqt (1|M0)         r4.1<1>:f     r4.9<0;1,0>:f                    {F@1}
(W)     add (4|M0)               r16.0<1>:f    r6.0<1;1,0>:f     r10.0<1;1,0>:f   {Compacted,I@1}
        mul (32|M0)              r8.0<1>:f     r30.0<1;1,0>:f    r1.10<0;1,0>:f   {Compacted}
(W)     add (1|M0)               r4.6<1>:f     r16.0<0;1,0>:f    r16.2<0;1,0>:f   {F@2}
(W)     add (1|M0)               r4.7<1>:f     r16.1<0;1,0>:f    r16.3<0;1,0>:f  
        mul (16|M0)              r18.0<1>:f    r7.0<1;1,0>:bf    r4.1<0;1,0>:f    {M@1}
        mul (16|M16)             r19.0<1>:f    r5.0<1;1,0>:bf    r4.1<0;1,0>:f   
(W)     add (1|M0)               r4.4<1>:f     r4.6<0;1,0>:f     r4.7<0;1,0>:f    {F@3}
        mul (32|M0)              acc0.0<1>:f   r8.0<1;1,0>:f     r18.0<1;1,0>:f   {Compacted,F@2}
(W)     add (1|M0)               r4.8<1>:f     r4.4<0;1,0>:f     1e-06:f               {F@2}
        mul (32|M0)              r60.0<1>:f    r32.0<1;1,0>:f    r1.10<0;1,0>:f   {Compacted}
        mul (16|M0)              r36.0<1>:f    r75.0<1;1,0>:bf   r4.1<0;1,0>:f   
        mul (16|M16)             r37.0<1>:f    r74.0<1;1,0>:bf   r4.1<0;1,0>:f   
        mul (16|M16)             r11.0<1>:f    r39.0<1;1,0>:bf   r4.1<0;1,0>:f   
        mul (16|M0)              r10.0<1>:f    r35.0<1;1,0>:bf   r4.1<0;1,0>:f   
        mul (16|M0)              r6.0<1>:f     r76.0<1;1,0>:bf   r4.1<0;1,0>:f   
        mul (16|M16)             r7.0<1>:f     r40.0<1;1,0>:bf   r4.1<0;1,0>:f   
        mul (32|M0)              acc2.0<1>:f   acc2.0<1;1,0>:f   r18.0<1;1,0>:f   {Compacted}
(W)     math.sqt (1|M0)          r4.1<1>:f     r4.8<0;1,0>:f                    {F@2}
        mul (32|M0)              r56.0<1>:f    r24.0<1;1,0>:f    r1.10<0;1,0>:f   {Compacted}
        mul (32|M0)              r64.0<1>:f    r28.0<1;1,0>:f    r1.10<0;1,0>:f   {Compacted}
        mad (32|M0)              acc0.0<1>:f   acc0.0<1;0>:f     r60.0<1;0>:f      r36.0<1>:f       {Compacted}
(W)     mul (1|M0)               r4.1<1>:f     r3.15<0;1,0>:f    r4.1<0;1,0>:f    {M@1}
        mad (32|M0)              r56.0<1>:f    acc2.0<1;0>:f     r56.0<1;0>:f      r36.0<1>:f       {Compacted,F@4}
        mad (32|M0)              r76.0<1>:f    acc0.0<1;0>:f     r64.0<1;0>:f      r10.0<1>:f       {Compacted,F@4}
(W)     mul (1|M0)               acc0.0<1>:d   r2.7<0;1,0>:d     r3.8<0;1,0>:uw   {F@1}
        mul (32|M0)              acc2.0<1>:f   r54.0<1;1,0>:f    r1.10<0;1,0>:f   {Compacted}
(W)     math.inv (1|M0)          r4.1<1>:f     r4.1<0;1,0>:f                   
(W)     macl (1|M0)              r5.0<1>:d     r2.7<0;1,0>:d     r3.4<0;1,0>:d   
        mul (32|M0)              acc0.0<1>:f   r68.0<1;1,0>:f    r1.10<0;1,0>:f   {Compacted,I@1}
        mul (32|M0)              acc2.0<1>:f   acc2.0<1;1,0>:f   r18.0<1;1,0>:f   {Compacted}
        mul (16|M0)              r34.0<1>:f    r34.0<1;1,0>:bf   r4.1<0;1,0>:f    {M@1}
        mul (16|M16)             r35.0<1>:f    r38.0<1;1,0>:bf   r4.1<0;1,0>:f   
        mul (16|M16)             r39.0<1>:f    r42.0<1;1,0>:bf   r4.1<0;1,0>:f   
        mul (16|M0)              r16.0<1>:f    r41.0<1;1,0>:bf   r4.1<0;1,0>:f   
        mul (16|M16)             r17.0<1>:f    r44.0<1;1,0>:bf   r4.1<0;1,0>:f   
        mul (16|M0)              r8.0<1>:f     r45.0<1;1,0>:bf   r4.1<0;1,0>:f   
        mul (16|M16)             r9.0<1>:f     r12.0<1;1,0>:bf   r4.1<0;1,0>:f   
        mul (16|M0)              r38.0<1>:f    r43.0<1;1,0>:bf   r4.1<0;1,0>:f   
        mul (32|M0)              r42.0<1>:f    r52.0<1;1,0>:f    r1.10<0;1,0>:f   {Compacted}
(W)     add (1|M0)               r4.1<1>:d     r5.0<0;1,0>:d     r3.5<0;1,0>:d    {Compacted,F@2}
        mul (32|M0)              acc0.0<1>:f   acc0.0<1;1,0>:f   r18.0<1;1,0>:f   {Compacted}
(W)     shl (1|M0)               r2.2<1>:q     r4.1<0;1,0>:d     1:w               {I@1}
        mad (32|M0)              acc2.0<1>:f   acc2.0<1;0>:f     r42.0<1;0>:f      r36.0<1>:f       {Compacted,F@2}
(W)     mov (2|M0)               r4.16<1>:w    0x40:uv                             
(W)     add (1|M0)               r42.0<1>:q    r2.2<0;1,0>:q     r1.3<0;1,0>:q    {Compacted,A@1}
        mul (32|M0)              r40.0<1>:f    r66.0<1;1,0>:f    r1.10<0;1,0>:f   {Compacted}
(W)     add (1|M0)               r12.0<1>:uq   r42.0<0;1,0>:uq   r4.16<0;1,0>:w   {I@1}
(W)     add (1|M0)               r12.1<1>:uq   r42.0<0;1,0>:uq   r4.17<0;1,0>:w  
        mul (32|M0)              r44.0<1>:f    r58.0<1;1,0>:f    r1.10<0;1,0>:f   {Compacted}
(W)     send.ugm (2|M0)          r5       r12  null:0  0x0            0x04100580           {I@1,$0} // wr:2+0, rd:1; load.ugm.d32.a64
        mul (32|M0)              r46.0<1>:f    r20.0<1;1,0>:f    r1.10<0;1,0>:f   {Compacted}
        mad (32|M0)              acc0.0<1>:f   acc0.0<1;0>:f     r40.0<1;0>:f      r36.0<1>:f       {Compacted,F@3}
        mul (32|M0)              r48.0<1>:f    r70.0<1;1,0>:f    r1.10<0;1,0>:f   {Compacted}
        mul (32|M0)              r78.0<1>:f    r26.0<1;1,0>:f    r1.10<0;1,0>:f   {Compacted}
        mad (32|M0)              acc2.0<1>:f   acc2.0<1;0>:f     r44.0<1;0>:f      r10.0<1>:f       {Compacted,F@5}
        mul (32|M0)              r74.0<1>:f    r50.0<1;1,0>:f    r1.10<0;1,0>:f   {Compacted}
        mul (32|M0)              r60.0<1>:f    r72.0<1;1,0>:f    r1.10<0;1,0>:f   {Compacted}
        mad (32|M0)              r56.0<1>:f    r56.0<1;0>:f      r46.0<1;0>:f      r10.0<1>:f       {Compacted,F@7}
        mul (32|M0)              r64.0<1>:f    r62.0<1;1,0>:f    r1.10<0;1,0>:f   {Compacted}
        mad (32|M0)              acc0.0<1>:f   acc0.0<1;0>:f     r48.0<1;0>:f      r10.0<1>:f       {Compacted,F@7}
(W)     mov (32|M0)              r46.0<1>:ud   0x0:ud                              {F@3}
(W)     mov (32|M0)              r40.0<1>:ud   0x0:ud                             
(W)     mov (32|M0)              r44.0<1>:ud   0x0:ud                             
        mad (32|M0)              r46.0<1>:f    r76.0<1;0>:f      r78.0<1;0>:f      r6.0<1>:f        {Compacted,I@3}
(W)     mov (32|M0)              r42.0<1>:ud   0x0:ud                             
        mad (32|M0)              r40.0<1>:f    acc0.0<1;0>:f     r60.0<1;0>:f      r6.0<1>:f        {Compacted,I@3}
        mad (32|M0)              r44.0<1>:f    r56.0<1;0>:f      r74.0<1;0>:f      r6.0<1>:f        {Compacted,I@2}
(W)     add (16|M0)              r12.0<1>:f    r46.0<1;1,0>:f    r47.0<1;1,0>:f   {Compacted,@3,$0.src}
        mad (32|M0)              r42.0<1>:f    acc2.0<1;0>:f     r64.0<1;0>:f      r6.0<1>:f        {Compacted,I@1}
(W)     add (16|M0)              r48.0<1>:f    r44.0<1;1,0>:f    r45.0<1;1,0>:f   {Compacted,F@3}
(W)     add (16|M0)              r46.0<1>:f    r40.0<1;1,0>:f    r41.0<1;1,0>:f   {Compacted}
(W)     mov (8|M0)               r49.0<1>:ud   r12.8<1;1,0>:ud                  {Compacted,F@4}
(W)     add (16|M0)              r47.0<1>:f    r42.0<1;1,0>:f    r43.0<1;1,0>:f   {Compacted,F@3}
(W)     mov (8|M0)               r56.0<1>:ud   r48.8<1;1,0>:ud                  {Compacted,F@3}
(W)     mov (8|M0)               r60.0<1>:ud   r46.8<1;1,0>:ud                  {Compacted,F@2}
(W)     add (8|M0)               r12.0<1>:f    r12.0<1;1,0>:f    r49.0<1;1,0>:f   {Compacted,I@3}
(W)     mov (8|M0)               r57.0<1>:ud   r47.8<1;1,0>:ud                  {Compacted,F@2}
(W)     add (8|M0)               r48.0<1>:f    r48.0<1;1,0>:f    r56.0<1;1,0>:f   {Compacted,I@3}
(W)     add (8|M0)               r46.0<1>:f    r46.0<1;1,0>:f    r60.0<1;1,0>:f   {Compacted,I@2}
(W)     mov (4|M0)               r49.0<1>:ud   r12.4<1;1,0>:ud                  {Compacted,F@3}
(W)     add (8|M0)               r47.0<1>:f    r47.0<1;1,0>:f    r57.0<1;1,0>:f   {Compacted,I@2}
(W)     mov (4|M0)               r56.0<1>:ud   r48.4<1;1,0>:ud                  {Compacted,F@3}
(W)     mov (4|M0)               r60.0<1>:ud   r46.4<1;1,0>:ud                  {Compacted,F@2}
(W)     add (4|M0)               r49.0<1>:f    r12.0<1;1,0>:f    r49.0<1;1,0>:f   {Compacted,I@3}
(W)     mov (4|M0)               r57.0<1>:ud   r47.4<1;1,0>:ud                  {Compacted,F@2}
(W)     add (4|M0)               r48.0<1>:f    r48.0<1;1,0>:f    r56.0<1;1,0>:f   {Compacted,I@3}
(W)     add (4|M0)               r12.0<1>:f    r46.0<1;1,0>:f    r60.0<1;1,0>:f   {Compacted,I@2}
(W)     add (1|M0)               r4.10<1>:f    r49.0<0;1,0>:f    r49.2<0;1,0>:f   {F@3}
(W)     add (1|M0)               r4.11<1>:f    r49.1<0;1,0>:f    r49.3<0;1,0>:f  
(W)     add (4|M0)               r47.0<1>:f    r47.0<1;1,0>:f    r57.0<1;1,0>:f   {Compacted,I@1}
(W)     add (1|M0)               r4.8<1>:f     r48.0<0;1,0>:f    r48.2<0;1,0>:f   {F@5}
(W)     add (1|M0)               r4.9<1>:f     r48.1<0;1,0>:f    r48.3<0;1,0>:f  
(W)     add (1|M0)               r4.4<1>:f     r12.0<0;1,0>:f    r12.2<0;1,0>:f   {Compacted,F@6}
(W)     add (1|M0)               r4.5<1>:f     r12.1<0;1,0>:f    r12.3<0;1,0>:f  
(W)     shl (1|M0)               r4.1<1>:ud    r5.0<0;1,0>:uw    0x10:uw              {$0.dst}
(W)     add (1|M0)               r4.12<1>:f    r4.10<0;1,0>:f    r4.11<0;1,0>:f   {F@6}
(W)     add (1|M0)               r4.6<1>:f     r47.0<0;1,0>:f    r47.2<0;1,0>:f   {F@6}
(W)     add (1|M0)               r4.7<1>:f     r47.1<0;1,0>:f    r47.3<0;1,0>:f  
(W)     add (1|M0)               r4.11<1>:f    r4.8<0;1,0>:f     r4.9<0;1,0>:f    {F@6}
(W)     add (1|M0)               r4.9<1>:f     r4.4<0;1,0>:f     r4.5<0;1,0>:f    {F@5}
(W)     shl (1|M0)               r4.4<1>:ud    r5.1<0;1,0>:uw    0x10:uw              {F@1}
(W)     add (1|M0)               r4.10<1>:f    r4.6<0;1,0>:f     r4.7<0;1,0>:f   
(W)     add (1|M0)               r4.6<1>:f     r4.1<0;1,0>:f     -r4.12<0;1,0>:f  {I@2}
(W)     shl (1|M0)               r4.8<1>:ud    r5.2<0;1,0>:uw    0x10:uw             
(W)     add (1|M0)               r4.5<1>:f     r4.4<0;1,0>:f     -r4.11<0;1,0>:f  {I@2}
(W)     shl (1|M0)               r4.7<1>:ud    r5.3<0;1,0>:uw    0x10:uw              {F@3}
(W)     mul (1|M0)               r2.9<1>:f     r4.6<0;1,0>:f     r2.6<0;1,0>:f    {F@2}
(W)     add (1|M0)               r4.4<1>:f     r4.8<0;1,0>:f     -r4.10<0;1,0>:f  {I@2}
(W)     mul (1|M0)               r2.8<1>:f     r4.5<0;1,0>:f     r2.6<0;1,0>:f    {F@3}
(W)     add (1|M0)               r4.1<1>:f     r4.7<0;1,0>:f     -r4.9<0;1,0>:f   {I@1}
        mul (32|M0)              acc0.0<1>:f   r18.0<1;1,0>:f    r2.9<0;1,0>:f    {Compacted,F@4}
(W)     mul (1|M0)               r2.7<1>:f     r4.4<0;1,0>:f     r2.6<0;1,0>:f    {F@4}
        mul (32|M0)              acc2.0<1>:f   r18.0<1;1,0>:f    r2.8<0;1,0>:f    {Compacted,F@4}
(W)     mul (1|M0)               r2.6<1>:f     r4.1<0;1,0>:f     r2.6<0;1,0>:f    {F@4}
        mul (32|M0)              r56.0<1>:f    r36.0<1;1,0>:f    r2.9<0;1,0>:f    {Compacted}
        mul (32|M0)              r46.0<1>:f    r18.0<1;1,0>:f    r2.7<0;1,0>:f    {Compacted,F@4}
        mad (32|M0)              r30.0<1>:f    acc0.0<1;0>:f     r30.0<1;0>:f      r1.10<0>:f       {Compacted}
        mad (32|M0)              r22.0<1>:f    acc2.0<1;0>:f     r22.0<1;0>:f      r1.10<0>:f       {Compacted}
        mul (32|M0)              r48.0<1>:f    r36.0<1;1,0>:f    r2.8<0;1,0>:f    {Compacted}
        mul (32|M0)              r18.0<1>:f    r18.0<1;1,0>:f    r2.6<0;1,0>:f    {Compacted,F@6}
        mul (32|M0)              acc2.0<1>:f   r36.0<1;1,0>:f    r2.7<0;1,0>:f    {Compacted}
        mul (32|M0)              r78.0<1>:f    r30.0<1;1,0>:f    r34.0<1;1,0>:f   {Compacted,F@5}
        mul (32|M0)              acc0.0<1>:f   r36.0<1;1,0>:f    r2.6<0;1,0>:f    {Compacted}
        mul (32|M0)              r76.0<1>:f    r10.0<1;1,0>:f    r2.9<0;1,0>:f    {Compacted}
        mul (32|M0)              r64.0<1>:f    r10.0<1;1,0>:f    r2.8<0;1,0>:f    {Compacted}
        mul (32|M0)              r80.0<1>:f    r10.0<1;1,0>:f    r2.7<0;1,0>:f    {Compacted}
        mad (32|M0)              r32.0<1>:f    r56.0<1;0>:f      r32.0<1;0>:f      r1.10<0>:f       {Compacted}
        mad (32|M0)              r54.0<1>:f    r46.0<1;0>:f      r54.0<1;0>:f      r1.10<0>:f       {Compacted}
        mul (32|M0)              r74.0<1>:f    r22.0<1;1,0>:f    r34.0<1;1,0>:f   {Compacted,F@7}
        mad (32|M0)              r24.0<1>:f    r48.0<1;0>:f      r24.0<1;0>:f      r1.10<0>:f       {Compacted,F@7}
        mad (32|M0)              r68.0<1>:f    r18.0<1;0>:f      r68.0<1;0>:f      r1.10<0>:f       {Compacted,F@7}
        mad (32|M0)              r52.0<1>:f    acc2.0<1;0>:f     r52.0<1;0>:f      r1.10<0>:f       {Compacted}
        mad (32|M0)              r66.0<1>:f    acc0.0<1;0>:f     r66.0<1;0>:f      r1.10<0>:f       {Compacted}
        mul (32|M0)              r10.0<1>:f    r10.0<1;1,0>:f    r2.6<0;1,0>:f    {Compacted}
        mul (32|M0)              r18.0<1>:f    r54.0<1;1,0>:f    r34.0<1;1,0>:f   {Compacted,F@7}
        mad (32|M0)              acc2.0<1>:f   r78.0<1;0>:f      r32.0<1;0>:f      r38.0<1>:f       {Compacted}
        mad (32|M0)              r28.0<1>:f    r76.0<1;0>:f      r28.0<1;0>:f      r1.10<0>:f       {Compacted}
        mul (32|M0)              r48.0<1>:f    r6.0<1;1,0>:f     r2.9<0;1,0>:f    {Compacted}
        mad (32|M0)              acc0.0<1>:f   r74.0<1;0>:f      r24.0<1;0>:f      r38.0<1>:f       {Compacted,F@7}
        mad (32|M0)              r20.0<1>:f    r64.0<1;0>:f      r20.0<1;0>:f      r1.10<0>:f       {Compacted}
        mul (32|M0)              r46.0<1>:f    r6.0<1;1,0>:f     r2.8<0;1,0>:f    {Compacted}
        mad (32|M0)              r70.0<1>:f    r10.0<1;0>:f      r70.0<1;0>:f      r1.10<0>:f       {Compacted,F@7}
        mad (32|M0)              r58.0<1>:f    r80.0<1;0>:f      r58.0<1;0>:f      r1.10<0>:f       {Compacted}
        mad (32|M0)              r18.0<1>:f    r18.0<1;0>:f      r52.0<1;0>:f      r38.0<1>:f       {Compacted,F@7}
        mad (32|M0)              r10.0<1>:f    acc2.0<1;0>:f     r28.0<1;0>:f      r16.0<1>:f       {Compacted,F@7}
        mad (32|M0)              r26.0<1>:f    r48.0<1;0>:f      r26.0<1;0>:f      r1.10<0>:f       {Compacted,F@7}
        mad (32|M0)              acc0.0<1>:f   acc0.0<1;0>:f     r20.0<1;0>:f      r16.0<1>:f       {Compacted,F@7}
(W)     mov (32|M0)              r44.0<1>:ud   0x0:ud                             
        mul (32|M0)              r60.0<1>:f    r68.0<1;1,0>:f    r34.0<1;1,0>:f   {Compacted}
        mad (32|M0)              r50.0<1>:f    r46.0<1;0>:f      r50.0<1;0>:f      r1.10<0>:f       {Compacted,F@7}
        mad (32|M0)              acc2.0<1>:f   r18.0<1;0>:f      r58.0<1;0>:f      r16.0<1>:f       {Compacted,F@6}
(W)     mov (32|M0)              r42.0<1>:ud   0x0:ud                             
        mad (32|M0)              r44.0<1>:f    r10.0<1;0>:f      r26.0<1;0>:f      r8.0<1>:f        {Compacted,A@2}
        mul (32|M0)              r36.0<1>:f    r6.0<1;1,0>:f     r2.7<0;1,0>:f    {Compacted}
        mul (32|M0)              r34.0<1>:f    r6.0<1;1,0>:f     r2.6<0;1,0>:f    {Compacted}
        mad (32|M0)              r42.0<1>:f    acc0.0<1;0>:f     r50.0<1;0>:f      r8.0<1>:f        {Compacted,A@1}
(W)     add (16|M0)              r10.0<1>:f    r44.0<1;1,0>:f    r45.0<1;1,0>:f   {Compacted,F@4}
        mad (32|M0)              r6.0<1>:f     r60.0<1;0>:f      r66.0<1;0>:f      r38.0<1>:f       {Compacted}
(W)     add (16|M0)              r5.0<1>:f     r42.0<1;1,0>:f    r43.0<1;1,0>:f   {Compacted,F@3}
(W)     mov (8|M0)               r11.0<1>:ud   r10.8<1;1,0>:ud                  {Compacted,F@3}
        mad (32|M0)              r6.0<1>:f     r6.0<1;0>:f       r70.0<1;0>:f      r16.0<1>:f       {Compacted,F@2}
(W)     mov (8|M0)               r16.0<1>:ud   r5.8<1;1,0>:ud                   {Compacted,F@1}
(W)     add (8|M0)               r10.0<1>:f    r10.0<1;1,0>:f    r11.0<1;1,0>:f   {Compacted,I@2}
        mad (32|M0)              r62.0<1>:f    r36.0<1;0>:f      r62.0<1;0>:f      r1.10<0>:f       {Compacted}
(W)     add (8|M0)               r5.0<1>:f     r5.0<1;1,0>:f     r16.0<1;1,0>:f   {Compacted,I@1}
(W)     mov (32|M0)              r40.0<1>:ud   0x0:ud                             
(W)     mov (4|M0)               r11.0<1>:ud   r10.4<1;1,0>:ud                  {Compacted,F@3}
(W)     mov (4|M0)               r16.0<1>:ud   r5.4<1;1,0>:ud                   {Compacted,F@1}
        mad (32|M0)              r40.0<1>:f    acc2.0<1;0>:f     r62.0<1;0>:f      r8.0<1>:f        {Compacted,I@3}
(W)     add (4|M0)               r10.0<1>:f    r10.0<1;1,0>:f    r11.0<1;1,0>:f   {Compacted,I@2}
        mad (32|M0)              r72.0<1>:f    r34.0<1;0>:f      r72.0<1;0>:f      r1.10<0>:f       {Compacted}
(W)     add (4|M0)               r5.0<1>:f     r5.0<1;1,0>:f     r16.0<1;1,0>:f   {Compacted,I@1}
(W)     add (16|M0)              r12.0<1>:f    r40.0<1;1,0>:f    r41.0<1;1,0>:f   {Compacted,F@4}
(W)     add (1|M0)               r4.6<1>:f     r10.0<0;1,0>:f    r10.2<0;1,0>:f   {F@4}
(W)     add (1|M0)               r4.7<1>:f     r10.1<0;1,0>:f    r10.3<0;1,0>:f  
(W)     mov (32|M0)              r10.0<1>:ud   0x0:ud                              {F@1}
(W)     add (1|M0)               r4.4<1>:f     r5.0<0;1,0>:f     r5.2<0;1,0>:f    {Compacted}
(W)     add (1|M0)               r4.5<1>:f     r5.1<0;1,0>:f     r5.3<0;1,0>:f   
        mad (32|M0)              r10.0<1>:f    r6.0<1;0>:f       r72.0<1;0>:f      r8.0<1>:f        {Compacted,I@1}
(W)     mov (8|M0)               r5.0<1>:ud    r12.8<1;1,0>:ud                  {Compacted,F@2}
(W)     add (1|M0)               r4.8<1>:f     r4.4<0;1,0>:f     r4.5<0;1,0>:f   
(W)     add (8|M0)               r6.0<1>:f     r12.0<1;1,0>:f    r5.0<1;1,0>:f    {Compacted,I@1}
(W)     add (16|M0)              r5.0<1>:f     r10.0<1;1,0>:f    r11.0<1;1,0>:f   {Compacted,F@3}
(W)     mov (4|M0)               r8.0<1>:ud    r6.4<1;1,0>:ud                   {Compacted,F@2}
(W)     mov (8|M0)               r7.0<1>:ud    r5.8<1;1,0>:ud                   {Compacted,F@1}
(W)     add (4|M0)               r6.0<1>:f     r6.0<1;1,0>:f     r8.0<1;1,0>:f    {Compacted,I@2}
(W)     add (8|M0)               r5.0<1>:f     r5.0<1;1,0>:f     r7.0<1;1,0>:f    {Compacted,I@1}
(W)     add (1|M0)               r4.4<1>:f     r6.0<0;1,0>:f     r6.2<0;1,0>:f    {Compacted,F@2}
(W)     add (1|M0)               r4.5<1>:f     r6.1<0;1,0>:f     r6.3<0;1,0>:f   
(W)     mov (4|M0)               r6.0<1>:ud    r5.4<1;1,0>:ud                   {Compacted,F@1}
(W)     add (1|M0)               r4.7<1>:f     r4.6<0;1,0>:f     r4.7<0;1,0>:f   
(W)     add (4|M0)               r5.0<1>:f     r5.0<1;1,0>:f     r6.0<1;1,0>:f    {Compacted,I@1}
(W)     add (1|M0)               r4.6<1>:f     r4.4<0;1,0>:f     r4.5<0;1,0>:f   
(W)     add (1|M0)               r4.4<1>:f     r5.0<0;1,0>:f     r5.2<0;1,0>:f    {Compacted,F@2}
(W)     add (1|M0)               r4.5<1>:f     r5.1<0;1,0>:f     r5.3<0;1,0>:f   
(W)     add (1|M0)               r4.1<1>:f     r4.4<0;1,0>:f     r4.5<0;1,0>:f    {F@1}
(~f3.0) goto (32|M0)                         L10912                  L10912                
L10776:
(W)     add (1|M0)               r10.0<1>:q    r2.2<0;1,0>:q     r1.0<0;1,0>:q    {Compacted}
(W)     mov (1|M0)               r6.1<1>:bf    r4.8<0;1,0>:f                   
(W)     mov (2|M0)               r4.16<1>:w    0x40:uv                              {F@1}
(W)     mov (1|M0)               r6.0<1>:bf    r4.7<0;1,0>:f                   
(W)     mov (1|M0)               r6.2<1>:bf    r4.6<0;1,0>:f                   
(W)     mov (1|M0)               r6.3<1>:bf    r4.1<0;1,0>:f                   
(W)     add (1|M0)               r8.0<1>:uq    r10.0<0;1,0>:uq   r4.16<0;1,0>:w   {I@1}
(W)     add (1|M0)               r8.1<1>:uq    r10.0<0;1,0>:uq   r4.17<0;1,0>:w  
(W)     send.ugm (2|M0)          null     r8  r6:1  0x0            0x04000584           {A@1,$6} // wr:2+1, rd:0; store.ugm.d32.a64
L10912:
        join (32|M0)                         L10928                                
L10928:
(W)     add (1|M0)               r1.12<1>:d    r1.12<0;1,0>:d    1:w              
(W)     cmp (32|M0)   (lt)f0.0   null<1>:d     r1.12<0;1,0>:d    r1.13<0;1,0>:d   {I@1}
(W&f0.0) jmpi                                L5832                                
L10976:
(W)     mul (1|M0)               acc0.0<1>:d   r3.6<0;1,0>:d     r3.4<0;1,0>:uw  
(W)     macl (1|M0)              r3.0<1>:d     r3.6<0;1,0>:d     r3.2<0;1,0>:d    {Compacted}
        mov (16|M16)             r2.16<1>:bf   r33.0<1;1,0>:f                  
(W)     mul (1|M0)               acc0.0<1>:d   r3.0<0;1,0>:d     r3.8<0;1,0>:uw   {I@1}
(W)     macl (1|M0)              r3.0<1>:d     r3.0<0;1,0>:d     r3.4<0;1,0>:d    {Compacted}
(W)     mul (1|M0)               acc0.0<1>:d   r3.5<0;1,0>:d     r3.4<0;1,0>:uw  
(W)     macl (1|M0)              r2.0<1>:d     r3.5<0;1,0>:d     r3.2<0;1,0>:d   
(W)     add (1|M0)               r3.1<1>:d     r3.0<0;1,0>:d     r2.0<0;1,0>:d    {Compacted,I@1}
        mov (16|M0)              r2.0<1>:bf    r32.0<1;1,0>:f                   {I@1}
        add (32|M0)              r8.0<1>:d     r14.0<1;1,0>:d    r3.1<0;1,0>:d    {Compacted,$6.src}
        mov (16|M0)              r5.0<2>:ud    r8.0<1;1,0>:ud                   {Compacted,I@1}
        shl (16|M0)              r11.0<1>:q    r5.0<2;1,0>:d     1:w               {I@1}
        mov (16|M16)             r5.0<2>:ud    r9.0<1;1,0>:ud                   {Compacted}
        shl (16|M16)             r9.0<1>:q     r5.0<2;1,0>:d     1:w               {I@1}
        add (16|M0)              r5.0<1>:q     r11.0<1;1,0>:q    r1.7<0;1,0>:q    {Compacted}
        add (16|M16)             r7.0<1>:q     r9.0<1;1,0>:q     r1.7<0;1,0>:q    {Compacted,I@2}
        mov (32|M0)              r10.0<1>:ud   r2.0<1;1,0>:uw                   {F@1}
        send.ugm (32|M0)         null     r5  r10:2  0x0            0x08000B84           {I@1,$1} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r8.0<1>:d     r82.0<1;1,0>:d    r3.1<0;1,0>:d    {Compacted,$1.src}
        mov (16|M0)              r5.0<2>:ud    r8.0<1;1,0>:ud                   {Compacted,I@1}
        shl (16|M0)              r11.0<1>:q    r5.0<2;1,0>:d     1:w               {I@1}
        mov (16|M16)             r5.0<2>:ud    r9.0<1;1,0>:ud                   {Compacted}
        mov (16|M0)              r2.0<1>:bf    r30.0<1;1,0>:f                  
        mov (16|M16)             r2.16<1>:bf   r31.0<1;1,0>:f                  
        shl (16|M16)             r9.0<1>:q     r5.0<2;1,0>:d     1:w               {I@1}
        add (16|M0)              r5.0<1>:q     r11.0<1;1,0>:q    r1.7<0;1,0>:q    {Compacted}
        add (16|M16)             r7.0<1>:q     r9.0<1;1,0>:q     r1.7<0;1,0>:q    {Compacted,I@2}
        mov (32|M0)              r10.0<1>:ud   r2.0<1;1,0>:uw                   {F@1}
        send.ugm (32|M0)         null     r5  r10:2  0x0            0x08000B84           {I@1,$2} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r8.0<1>:d     r84.0<1;1,0>:d    r3.1<0;1,0>:d    {Compacted,$2.src}
        mov (16|M0)              r5.0<2>:ud    r8.0<1;1,0>:ud                   {Compacted,I@1}
        shl (16|M0)              r11.0<1>:q    r5.0<2;1,0>:d     1:w               {I@1}
        mov (16|M16)             r5.0<2>:ud    r9.0<1;1,0>:ud                   {Compacted}
        mov (16|M0)              r2.0<1>:bf    r28.0<1;1,0>:f                  
        mov (16|M16)             r2.16<1>:bf   r29.0<1;1,0>:f                  
        shl (16|M16)             r9.0<1>:q     r5.0<2;1,0>:d     1:w               {I@1}
        add (16|M0)              r5.0<1>:q     r11.0<1;1,0>:q    r1.7<0;1,0>:q    {Compacted}
        add (16|M16)             r7.0<1>:q     r9.0<1;1,0>:q     r1.7<0;1,0>:q    {Compacted,I@2}
        mov (32|M0)              r10.0<1>:ud   r2.0<1;1,0>:uw                   {F@1}
        send.ugm (32|M0)         null     r5  r10:2  0x0            0x08000B84           {I@1,$3} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r8.0<1>:d     r86.0<1;1,0>:d    r3.1<0;1,0>:d    {Compacted,$3.src}
        mov (16|M0)              r5.0<2>:ud    r8.0<1;1,0>:ud                   {Compacted,I@1}
        shl (16|M0)              r11.0<1>:q    r5.0<2;1,0>:d     1:w               {I@1}
        mov (16|M16)             r5.0<2>:ud    r9.0<1;1,0>:ud                   {Compacted}
        mov (16|M0)              r2.0<1>:bf    r26.0<1;1,0>:f                  
        mov (16|M16)             r2.16<1>:bf   r27.0<1;1,0>:f                  
(W)     or (1|M0)                r3.1<1>:d     r3.5<0;1,0>:d     1:w              
        shl (16|M16)             r9.0<1>:q     r5.0<2;1,0>:d     1:w               {I@2}
(W)     mul (1|M0)               acc0.0<1>:d   r3.1<0;1,0>:d     r3.4<0;1,0>:uw   {I@2}
        add (16|M0)              r5.0<1>:q     r11.0<1;1,0>:q    r1.7<0;1,0>:q    {Compacted}
        add (16|M16)             r7.0<1>:q     r9.0<1;1,0>:q     r1.7<0;1,0>:q    {Compacted,I@3}
        mov (32|M0)              r10.0<1>:ud   r2.0<1;1,0>:uw                   {F@1}
(W)     macl (1|M0)              r2.0<1>:d     r3.1<0;1,0>:d     r3.2<0;1,0>:d    {Compacted}
        send.ugm (32|M0)         null     r5  r10:2  0x0            0x08000B84           {I@2,$4} // wr:4+2, rd:0; store.ugm.d16u32.a64
(W)     add (1|M0)               r3.1<1>:d     r3.0<0;1,0>:d     r2.0<0;1,0>:d    {Compacted,I@1}
        mov (16|M16)             r2.16<1>:bf   r25.0<1;1,0>:f                  
        add (32|M0)              r8.0<1>:d     r14.0<1;1,0>:d    r3.1<0;1,0>:d    {Compacted,@1,$4.src}
        mov (16|M0)              r5.0<2>:ud    r8.0<1;1,0>:ud                   {Compacted,I@1}
        shl (16|M0)              r11.0<1>:q    r5.0<2;1,0>:d     1:w               {I@1}
        mov (16|M16)             r5.0<2>:ud    r9.0<1;1,0>:ud                   {Compacted}
        mov (16|M0)              r2.0<1>:bf    r24.0<1;1,0>:f                  
        shl (16|M16)             r9.0<1>:q     r5.0<2;1,0>:d     1:w               {I@1}
        add (16|M0)              r5.0<1>:q     r11.0<1;1,0>:q    r1.7<0;1,0>:q    {Compacted}
        add (16|M16)             r7.0<1>:q     r9.0<1;1,0>:q     r1.7<0;1,0>:q    {Compacted,I@2}
        mov (32|M0)              r10.0<1>:ud   r2.0<1;1,0>:uw                   {F@1}
        send.ugm (32|M0)         null     r5  r10:2  0x0            0x08000B84           {I@1,$5} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r8.0<1>:d     r82.0<1;1,0>:d    r3.1<0;1,0>:d    {Compacted,$5.src}
        mov (16|M0)              r5.0<2>:ud    r8.0<1;1,0>:ud                   {Compacted,I@1}
        shl (16|M0)              r11.0<1>:q    r5.0<2;1,0>:d     1:w               {I@1}
        mov (16|M16)             r5.0<2>:ud    r9.0<1;1,0>:ud                   {Compacted}
        mov (16|M0)              r2.0<1>:bf    r22.0<1;1,0>:f                  
        mov (16|M16)             r2.16<1>:bf   r23.0<1;1,0>:f                  
        shl (16|M16)             r9.0<1>:q     r5.0<2;1,0>:d     1:w               {I@1}
        add (16|M0)              r5.0<1>:q     r11.0<1;1,0>:q    r1.7<0;1,0>:q    {Compacted}
        add (16|M16)             r7.0<1>:q     r9.0<1;1,0>:q     r1.7<0;1,0>:q    {Compacted,I@2}
        mov (32|M0)              r10.0<1>:ud   r2.0<1;1,0>:uw                   {F@1}
        send.ugm (32|M0)         null     r5  r10:2  0x0            0x08000B84           {I@1,$6} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r8.0<1>:d     r84.0<1;1,0>:d    r3.1<0;1,0>:d    {Compacted,$6.src}
        mov (16|M0)              r5.0<2>:ud    r8.0<1;1,0>:ud                   {Compacted,I@1}
        shl (16|M0)              r11.0<1>:q    r5.0<2;1,0>:d     1:w               {I@1}
        mov (16|M16)             r5.0<2>:ud    r9.0<1;1,0>:ud                   {Compacted}
        mov (16|M0)              r2.0<1>:bf    r20.0<1;1,0>:f                  
        mov (16|M16)             r2.16<1>:bf   r21.0<1;1,0>:f                  
        shl (16|M16)             r9.0<1>:q     r5.0<2;1,0>:d     1:w               {I@1}
        add (16|M0)              r5.0<1>:q     r11.0<1;1,0>:q    r1.7<0;1,0>:q    {Compacted}
        add (16|M16)             r7.0<1>:q     r9.0<1;1,0>:q     r1.7<0;1,0>:q    {Compacted,I@2}
        mov (32|M0)              r10.0<1>:ud   r2.0<1;1,0>:uw                   {F@1}
        send.ugm (32|M0)         null     r5  r10:2  0x0            0x08000B84           {I@1,$7} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r8.0<1>:d     r86.0<1;1,0>:d    r3.1<0;1,0>:d    {Compacted,$7.src}
        mov (16|M0)              r5.0<2>:ud    r8.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M0)              r2.0<1>:bf    r50.0<1;1,0>:f                  
        mov (16|M16)             r2.16<1>:bf   r51.0<1;1,0>:f                  
(W)     or (1|M0)                r3.1<1>:d     r3.5<0;1,0>:d     2:w              
        shl (16|M0)              r18.0<1>:q    r5.0<2;1,0>:d     1:w               {I@2}
        mov (16|M16)             r5.0<2>:ud    r9.0<1;1,0>:ud                   {Compacted}
(W)     mul (1|M0)               acc0.0<1>:d   r3.1<0;1,0>:d     r3.4<0;1,0>:uw   {I@3}
        shl (16|M16)             r16.0<1>:q    r5.0<2;1,0>:d     1:w               {I@2}
        mov (32|M0)              r6.0<1>:ud    r2.0<1;1,0>:uw                   {F@1}
(W)     macl (1|M0)              r2.0<1>:d     r3.1<0;1,0>:d     r3.2<0;1,0>:d    {Compacted}
        add (16|M0)              r8.0<1>:q     r18.0<1;1,0>:q    r1.7<0;1,0>:q    {Compacted}
        add (16|M16)             r10.0<1>:q    r16.0<1;1,0>:q    r1.7<0;1,0>:q    {Compacted,I@4}
(W)     add (1|M0)               r3.1<1>:d     r3.0<0;1,0>:d     r2.0<0;1,0>:d    {Compacted,I@3}
        send.ugm (32|M0)         null     r8  r6:2  0x0            0x08000B84           {I@2,$8} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r8.0<1>:d     r14.0<1;1,0>:d    r3.1<0;1,0>:d    {Compacted,@1,$8.src}
        mov (16|M0)              r5.0<2>:ud    r8.0<1;1,0>:ud                   {Compacted,I@1}
        shl (16|M0)              r18.0<1>:q    r5.0<2;1,0>:d     1:w               {I@1}
        mov (16|M16)             r5.0<2>:ud    r9.0<1;1,0>:ud                   {Compacted}
        mov (16|M16)             r2.16<1>:bf   r53.0<1;1,0>:f                  
        mov (16|M0)              r2.0<1>:bf    r52.0<1;1,0>:f                  
        shl (16|M16)             r16.0<1>:q    r5.0<2;1,0>:d     1:w               {I@1}
        add (16|M0)              r8.0<1>:q     r18.0<1;1,0>:q    r1.7<0;1,0>:q    {Compacted}
        add (16|M16)             r10.0<1>:q    r16.0<1;1,0>:q    r1.7<0;1,0>:q    {Compacted,I@2}
        mov (32|M0)              r6.0<1>:ud    r2.0<1;1,0>:uw                   {F@1}
        send.ugm (32|M0)         null     r8  r6:2  0x0            0x08000B84           {I@1,$9} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r8.0<1>:d     r82.0<1;1,0>:d    r3.1<0;1,0>:d    {Compacted,$9.src}
        mov (16|M0)              r5.0<2>:ud    r8.0<1;1,0>:ud                   {Compacted,I@1}
        shl (16|M0)              r18.0<1>:q    r5.0<2;1,0>:d     1:w               {I@1}
        mov (16|M16)             r5.0<2>:ud    r9.0<1;1,0>:ud                   {Compacted}
        mov (16|M0)              r2.0<1>:bf    r54.0<1;1,0>:f                  
        mov (16|M16)             r2.16<1>:bf   r55.0<1;1,0>:f                  
        shl (16|M16)             r16.0<1>:q    r5.0<2;1,0>:d     1:w               {I@1}
        add (16|M0)              r8.0<1>:q     r18.0<1;1,0>:q    r1.7<0;1,0>:q    {Compacted}
        add (16|M16)             r10.0<1>:q    r16.0<1;1,0>:q    r1.7<0;1,0>:q    {Compacted,I@2}
        mov (32|M0)              r6.0<1>:ud    r2.0<1;1,0>:uw                   {F@1}
        send.ugm (32|M0)         null     r8  r6:2  0x0            0x08000B84           {I@1,$10} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r8.0<1>:d     r84.0<1;1,0>:d    r3.1<0;1,0>:d    {Compacted,$10.src}
        mov (16|M0)              r5.0<2>:ud    r8.0<1;1,0>:ud                   {Compacted,I@1}
        shl (16|M0)              r18.0<1>:q    r5.0<2;1,0>:d     1:w               {I@1}
        mov (16|M16)             r5.0<2>:ud    r9.0<1;1,0>:ud                   {Compacted}
        mov (16|M0)              r2.0<1>:bf    r58.0<1;1,0>:f                  
        mov (16|M16)             r2.16<1>:bf   r59.0<1;1,0>:f                  
        shl (16|M16)             r16.0<1>:q    r5.0<2;1,0>:d     1:w               {I@1}
        add (16|M0)              r8.0<1>:q     r18.0<1;1,0>:q    r1.7<0;1,0>:q    {Compacted}
        add (16|M16)             r10.0<1>:q    r16.0<1;1,0>:q    r1.7<0;1,0>:q    {Compacted,I@2}
        mov (32|M0)              r6.0<1>:ud    r2.0<1;1,0>:uw                   {F@1}
        send.ugm (32|M0)         null     r8  r6:2  0x0            0x08000B84           {I@1,$11} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r8.0<1>:d     r86.0<1;1,0>:d    r3.1<0;1,0>:d    {Compacted,$11.src}
        mov (16|M0)              r5.0<2>:ud    r8.0<1;1,0>:ud                   {Compacted,I@1}
        mov (16|M0)              r2.0<1>:bf    r62.0<1;1,0>:f                  
        mov (16|M16)             r2.16<1>:bf   r63.0<1;1,0>:f                  
(W)     or (1|M0)                r3.1<1>:d     r3.5<0;1,0>:d     3:w              
        shl (16|M0)              r18.0<1>:q    r5.0<2;1,0>:d     1:w               {I@2}
        mov (16|M16)             r5.0<2>:ud    r9.0<1;1,0>:ud                   {Compacted}
(W)     mul (1|M0)               acc0.0<1>:d   r3.1<0;1,0>:d     r3.4<0;1,0>:uw   {I@3}
        shl (16|M16)             r16.0<1>:q    r5.0<2;1,0>:d     1:w               {I@2}
        mov (32|M0)              r6.0<1>:ud    r2.0<1;1,0>:uw                   {F@1}
(W)     macl (1|M0)              r2.0<1>:d     r3.1<0;1,0>:d     r3.2<0;1,0>:d    {Compacted}
        add (16|M0)              r8.0<1>:q     r18.0<1;1,0>:q    r1.7<0;1,0>:q    {Compacted}
        add (16|M16)             r10.0<1>:q    r16.0<1;1,0>:q    r1.7<0;1,0>:q    {Compacted,I@4}
(W)     add (1|M0)               r4.1<1>:d     r3.0<0;1,0>:d     r2.0<0;1,0>:d    {Compacted,I@3}
        send.ugm (32|M0)         null     r8  r6:2  0x0            0x08000B84           {I@2,$12} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r6.0<1>:d     r14.0<1;1,0>:d    r4.1<0;1,0>:d    {Compacted,@1,$12.src}
        mov (16|M0)              r2.0<2>:ud    r6.0<1;1,0>:ud                   {Compacted,I@1}
        shl (16|M0)              r14.0<1>:q    r2.0<2;1,0>:d     1:w               {I@1}
        mov (16|M16)             r2.0<2>:ud    r7.0<1;1,0>:ud                   {Compacted}
        mov (16|M0)              r9.0<1>:bf    r66.0<1;1,0>:f                  
        mov (16|M16)             r9.16<1>:bf   r67.0<1;1,0>:f                  
        shl (16|M16)             r10.0<1>:q    r2.0<2;1,0>:d     1:w               {I@1}
        add (16|M0)              r5.0<1>:q     r14.0<1;1,0>:q    r1.7<0;1,0>:q    {Compacted}
        add (16|M16)             r7.0<1>:q     r10.0<1;1,0>:q    r1.7<0;1,0>:q    {Compacted,I@2}
        mov (32|M0)              r2.0<1>:ud    r9.0<1;1,0>:uw                   {F@1}
        send.ugm (32|M0)         null     r5  r2:2  0x0            0x08000B84           {I@1,$13} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r6.0<1>:d     r82.0<1;1,0>:d    r4.1<0;1,0>:d    {Compacted,$13.src}
        mov (16|M0)              r2.0<2>:ud    r6.0<1;1,0>:ud                   {Compacted,I@1}
        shl (16|M0)              r14.0<1>:q    r2.0<2;1,0>:d     1:w               {I@1}
        mov (16|M16)             r2.0<2>:ud    r7.0<1;1,0>:ud                   {Compacted}
        mov (16|M0)              r11.0<1>:bf   r68.0<1;1,0>:f                  
        mov (16|M16)             r11.16<1>:bf  r69.0<1;1,0>:f                  
        shl (16|M16)             r9.0<1>:q     r2.0<2;1,0>:d     1:w               {I@1}
        add (16|M0)              r5.0<1>:q     r14.0<1;1,0>:q    r1.7<0;1,0>:q    {Compacted}
        add (16|M16)             r7.0<1>:q     r9.0<1;1,0>:q     r1.7<0;1,0>:q    {Compacted,I@2}
        mov (32|M0)              r2.0<1>:ud    r11.0<1;1,0>:uw                  {F@1}
        send.ugm (32|M0)         null     r5  r2:2  0x0            0x08000B84           {I@1,$14} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r6.0<1>:d     r84.0<1;1,0>:d    r4.1<0;1,0>:d    {Compacted,$14.src}
        mov (16|M0)              r2.0<2>:ud    r6.0<1;1,0>:ud                   {Compacted,I@1}
        shl (16|M0)              r14.0<1>:q    r2.0<2;1,0>:d     1:w               {I@1}
        mov (16|M16)             r2.0<2>:ud    r7.0<1;1,0>:ud                   {Compacted}
        mov (16|M0)              r11.0<1>:bf   r70.0<1;1,0>:f                  
        mov (16|M16)             r11.16<1>:bf  r71.0<1;1,0>:f                  
        shl (16|M16)             r9.0<1>:q     r2.0<2;1,0>:d     1:w               {I@1}
        add (16|M0)              r5.0<1>:q     r14.0<1;1,0>:q    r1.7<0;1,0>:q    {Compacted}
        add (16|M16)             r7.0<1>:q     r9.0<1;1,0>:q     r1.7<0;1,0>:q    {Compacted,I@2}
        mov (32|M0)              r2.0<1>:ud    r11.0<1;1,0>:uw                  {F@1}
        send.ugm (32|M0)         null     r5  r2:2  0x0            0x08000B84           {I@1,$15} // wr:4+2, rd:0; store.ugm.d16u32.a64
        add (32|M0)              r6.0<1>:d     r86.0<1;1,0>:d    r4.1<0;1,0>:d    {Compacted,$15.src}
        mov (16|M0)              r2.0<2>:ud    r6.0<1;1,0>:ud                   {Compacted,I@1}
        shl (16|M0)              r11.0<1>:q    r2.0<2;1,0>:d     1:w               {I@1}
        mov (16|M16)             r2.0<2>:ud    r7.0<1;1,0>:ud                   {Compacted}
        mov (16|M0)              r14.0<1>:bf   r72.0<1;1,0>:f                  
        mov (16|M16)             r14.16<1>:bf  r73.0<1;1,0>:f                  
        shl (16|M16)             r9.0<1>:q     r2.0<2;1,0>:d     1:w               {I@1}
        add (16|M0)              r5.0<1>:q     r11.0<1;1,0>:q    r1.7<0;1,0>:q    {Compacted}
        add (16|M16)             r7.0<1>:q     r9.0<1;1,0>:q     r1.7<0;1,0>:q    {Compacted,I@2}
        mov (32|M0)              r2.0<1>:ud    r14.0<1;1,0>:uw                  {F@1}
        send.ugm (32|M0)         null     r5  r2:2  0x0            0x08000B84           {I@1,$0} // wr:4+2, rd:0; store.ugm.d16u32.a64
L13384:
(W)     mov (16|M0)              r112.0<1>:f   r13.0<1;1,0>:f                   {Compacted}
(W)     send.gtwy (1|M0)         null     r112  null:0  0x0            0x02000010           {EOT,F@1,$1} // wr:1+0, rd:0; end of thread
L13408:
(W)     mov (16|M0)              null<1>:ud    0xBA0B4088:ud                             
(W)     mov (16|M0)              null<1>:ud    0xAC922C90:ud                             
(W)     mov (16|M0)              null<1>:ud    0x0:ud                             
(W)     mov (16|M0)              null<1>:ud    0x2D:ud                             
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
