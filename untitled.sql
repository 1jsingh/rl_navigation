set mapreduce.map.memory.mb=12000;
set mapreduce.map.java.opts=-Xmx7200m;
set mapreduce.reduce.memory.mb=12000;
set mapreduce.reduce.java.opts=-Xmx7200m;

DROP VIEW t1;
CREATE VIEW t1 as
SELECT sentences(user_agent) as ua,
-- sentences(advertisers[0]["i_ctgry"]) as i_ctgry,
partners[0]["src"] as src_id,
partners[0]["device"] as device_type,
advertisers[0]["c_price"] as c_price,
advertisers[0]["nctr"] as ctr,
lcookie
FROM ydn_feed.impression
WHERE yyyymmdd = '20180801';

DROP VIEW t2;
CREATE VIEW t2 as
SELECT CASE WHEN ua[0][3] = "Android" THEN "Android"
WHEN ua[0][2] = "iPhone" THEN "iPhone"
ELSE NULL
END as os_type,
CASE WHEN LENGTH(lcookie) <> 0 THEN "lcookie"
ELSE "no_lcookie"
END as lid,
CAST(c_price as float) as c_price,
CAST(ctr as float) as ctr
FROM t1
WHERE device_type = 4
AND src_id IN (332066,332067);

SELECT os_type, lid, 
CORR(ctr,c_price)
FROM t2
GROUP BY os_type,lid
ORDER BY os_type,lid;