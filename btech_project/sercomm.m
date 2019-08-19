function sercomm(data)

sobj=serial('COM1');
fopen(sobj);
fprintf(sobj,data);
fprintf(data);
fclose(sobj);
delete(sobj);