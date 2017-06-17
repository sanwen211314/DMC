function  [X] = r2zDMC(X_dec,d_dec,minGrade,maxGrade,Uob)

          X = zeros(size(X_dec));
            for i = 1:length(Uob)
                t = floor(X_dec(Uob(i)));
                if t < minGrade
                    X(Uob(i)) = minGrade;
                elseif t >= maxGrade
                    X(Uob(i)) = maxGrade;
                elseif X_dec(Uob(i)) - t - d_dec(t+1) <= 0
                    X(Uob(i)) = t;
                else
                    X(Uob(i)) = t + 1;
                end
            end
end
