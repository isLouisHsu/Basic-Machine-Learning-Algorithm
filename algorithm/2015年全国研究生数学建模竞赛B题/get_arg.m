function c = get_arg(a, b)
    c = acos(dot(a,b)/(norm(a)*norm(b))) * 180 / pi;
end
